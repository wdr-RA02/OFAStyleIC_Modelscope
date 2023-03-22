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
        self.reward_calc=RewardCalculator()
        super().__init__()


    def forward(self, model, inputs):
        assert isinstance(model, OfaForAllTasks), \
            "model required to be OfaForAllTasks type, got {}".format(type(model))
        
        assert id(model.tokenizer)==id(self.tokenizer), \
            "tokenizers from model and trainer don't match, this is usually unexpected. "
        '''
        gen_tgt_tokens: decode后的sampled_sequence [Tensor(n_words) * batch_size]
        gen_tgt_words: sampled_sequence的token  [str * batch_size]
        gt_words:  decode后的ground truth    [str * batch_size]
        '''
        # Step1: generate sample and baseline
        gen_tgt_tokens, gen_tgt_words, \
        baseline_tokens, baseline_words = self.get_sample_from_beams(model, inputs)
        
        # Step2: calculate rewards
        gt_batch=inputs["labels"]
        reward_sample=self.reward_calc(gen_tgt_words, gt_batch)
        baseline_sample=self.reward_calc(baseline_words, gt_batch)

        
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
    
    # Step2: generate rewards
    def calculate_reward(self, gen_res, greedy_res, ground_truth):
        pass