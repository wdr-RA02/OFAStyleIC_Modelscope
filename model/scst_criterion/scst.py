import random

import numpy as np
import torch
import torch.nn.functional as F
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.utils.config import Config
from torch.nn.modules.loss import _Loss
from .reward_calc import RewardCalculator


class SelfCriticalSeqTrainingCriterion(_Loss):
    def __init__(self, args:Config) -> None:
        self.cfg=args
        self.tokenizer=args.tokenizer
        # padding
        self.padding_idx = args.tokenizer.pad_token_id
        self.bos_idx=args.tokenizer.bos_token_id
        self.reward_calc=RewardCalculator(eos_token=args.tokenizer.eos_token)
        super().__init__()


    def forward(self, model, inputs):
        assert isinstance(model, OfaForAllTasks), \
            "model required to be OfaForAllTasks type, got {}".format(type(model))
        
        assert id(model.tokenizer)==id(self.tokenizer), \
            "tokenizers from model and trainer don't match, which is usually unexpected. "
        '''
        gen_tgt_tokens: decode后的sampled_sequence [Tensor(n_words) * batch_size]
        gen_tgt_words: sampled_sequence的token  [str * batch_size]
        gt_words:  decode后的ground truth    [str * batch_size]
        '''
        # Step1: generate sample and baseline
        self.device=model.model.device

        gen_tgt_tokens, gen_tgt_words, \
        baseline_tokens, baseline_words = self.get_sample_from_beams(model, inputs)
        
        # Step2: calculate rewards
        gt_batch=inputs["labels"]
        # reward_sample, rewards_sample=self.reward_calc(gen_tgt_words, gt_batch)
        # reward_baseline, rewards_baseline=self.reward_calc(baseline_words, gt_batch)
        reference={"sampled": gen_tgt_words, "greedy": baseline_words}
        rel_reward, rel_rewards=self.reward_calc(reference, gt_batch)

        # Step3: get gradient
        # Step3.1: get model output using the sample input
        model_output, target_tokens=self.get_output_of_model(model, inputs, gen_tgt_tokens)
        # Step3.2: get log probability
        log_prob=self.get_logprob(model_output["logits"])
        # Step3.3: get loss
        # can't believe this is a numpy array...
        rel_rewards_=torch.asarray(rel_rewards, device=self.device)
        loss, ntokens=self.calculate_scst_loss(log_prob, rel_rewards_, target_tokens)

        loss_data=loss.sum()
        logging_output={
            "loss": loss_data,
            "score": rel_rewards_.sum(),
            "ntokens": ntokens,
            "sample_size": ntokens
        }
        return loss_data, ntokens, logging_output

        
    def get_sample_from_beams(self, model, inputs):
        '''
        从model的inference输出(beam_size个)随机sample一个作为sampled sequence

        args:
        model: OfaModelForAll
        inputs: train preprocessor的输出
        
        return:
        gen_target: sampled_sequence的token list
        gen_res: decode后的sampled_sequence list
        baseline_res:  decode后的baseline list, 使用贪婪解码
        '''
        model.model.eval()
        with torch.no_grad():
            # generate the candidates with beam decoding
            beam_candidates=model.generator.generate([model.model],
                                              inputs,
                                              prefix_tokens=inputs.get(
                                                  'prefix_tokens', None))
            greedy_decode=self.greedy_decode(model, inputs)

        gen_target=[]
        gen_res=[]
        greedy_target=[]
        greedy_res=[]
        # beam_size=5
        for i in range(len(beam_candidates)):
            # randomly pick the candidate
            # len(gen_candidates)==batch_size
            lucky_cap_id=random.randint(0, len(beam_candidates[i])-1)
            lucky_cap=beam_candidates[i][lucky_cap_id]["tokens"]
            # save token
            gen_target.append(lucky_cap.int())
            # save decoded gen and ground truth
            gen_res.append(self.tokenizer.decode(lucky_cap).strip())
        
        for i in range(len(greedy_decode)):
            # we can securely access via [0]
            greedy_cap=greedy_decode[i][0]["tokens"]

            greedy_target.append(greedy_cap.int())
            greedy_res.append(self.tokenizer.decode(greedy_cap).strip())


        # gen_target: List[Tensor]
        # gen_res: List[str]
        return gen_target, gen_res, greedy_target, greedy_res
    
    @torch.no_grad()
    def greedy_decode(self, model, inputs):
        '''
        对当前input使用greedy decode得到一个baseline sequence
        '''
        # 先偷个懒借用一下beam=1, 有时间再改(bushi)
        model.model.eval()
        beam=model.generator.beam_size
        # when beam_size=1, the beam search reduces to greedy
        model.generator.beam_size=1
        greedy_baseline=model.generator.generate([model.model],
                                              inputs,
                                              prefix_tokens=inputs.get(
                                                  'prefix_tokens', None))
        # return the beam
        model.generator.beam_size=beam
        # print(logits.shape)
        return greedy_baseline
    
    def get_output_of_model(self, model, inputs, sample_tokens):
        '''
        将sample sequence作为encoder input, 得到logits等等输出项

        注: sample_tokens含有eos

        return: 
        model_output
        new_target: 含有padding的二维sample_token Tensor, size=[b x max_word]
        '''

        model.model.train()
        batch_size=inputs["nsentences"]
        net_input=inputs["net_input"]

        # replace net_input["decoder_input_ids"] to that of the sample
        # insert each token list to new_dec_input_ids[i, :] and pad with padding_idx
        seq_length=max(v.shape[0] for v in sample_tokens)
        # seq len+1 for bos
        new_dec_input_ids=torch.full(size=(batch_size, seq_length+1), 
                                     device=self.device,
                                     dtype=torch.int64,
                                     fill_value=self.padding_idx)
        new_target=torch.full_like(new_dec_input_ids, fill_value=self.padding_idx)

        # fill bos in the first column
        new_dec_input_ids[:, 0].fill_(self.bos_idx)

        for i, one_sample_token in enumerate(sample_tokens):
            # fill in the tokens by line
            new_dec_input_ids[i, 1:len(one_sample_token)].copy_(one_sample_token[:-1])
            new_target[i, 0:len(one_sample_token)].copy_(one_sample_token)


        net_input.update({"decoder_input_ids": new_dec_input_ids})
        inputs.update({"target": new_target})
        model_output=model.model(**net_input)

        return model_output, new_target
    
    def get_logprob(self, logits):
        '''
        从模型输出中得到log probability

        args: logits: net_output["logits"]

        return: log probability
        '''
        log_prob=F.log_softmax(logits, dim=-1, dtype=torch.float32)

        return log_prob

    
    def calculate_scst_loss(self, log_prob, scores, target):
        '''
        根据scst公式计算loss

        args:
        log_prob: 使用get_logprob得到的log probability, shape=[b, n, vocab_size]
        scores: r(w^s)-r(\hat{w}), shape=[b,]
        target: sample seq的token矩阵, shape=[b,n]

        return:
        loss: scst loss of the seq
        ntokens: token的总数目
        '''
        assert len(log_prob.shape)==3, \
            "log_prob is supposed to be [b, n, vocab], got {}".format(log_prob.shape)
        log_prob_for_each_word=log_prob.gather(dim=-1, index=target.unsqueeze(-1)).squeeze()
        # 获得的是log p(w_t|h, w_1~w_{t-1}), shape=[b, n:=words_in_seq]
        # 解释可以见20230322的log

        loss=(-log_prob_for_each_word.sum(dim=-1)*scores)
        # \nabla=(r(sample)-r(greedy)) \nabla \sigma{log(p(w_t|h, w_t-1))}

        ntokens=target.numel()
        return loss, ntokens

        