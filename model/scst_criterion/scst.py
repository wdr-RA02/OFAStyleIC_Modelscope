import random

import numpy as np
import torch
import torch.nn.functional as F
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.utils.config import Config
from torch.nn.modules.loss import _Loss


class SelfCriticalSeqTrainingCriterion(_Loss):
    def __init__(self, args:Config) -> None:
        self.cfg=args
        self.tokenizer=args.tokenizer
        # padding
        self.padding_idx = args.tokenizer.pad_token_id
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
        gen_tgt_tokens, gen_tgt_words, gt_words=self.get_sample_from_beams(model, inputs)

    def get_sample_from_beams(self, model, inputs):
        '''
        从model的inference输出(beam_size个)随机sample一个作为sampled sequence

        args:
        model: OfaModelForAll
        inputs: train preprocessor的输出
        
        return:
        gen_target: sampled_sequence的token list
        gen_res: decode后的sampled_sequence list
        gt_res:  decode后的ground truth list
        '''
        model.model.eval()
        with torch.no_grad():
            # generate the candidates with beam decoding
            gen_candidates=model.generator.generate([model.model],
                                              inputs,
                                              prefix_tokens=inputs.get(
                                                  'prefix_tokens', None))
        gen_target=[]
        gen_res=[]
        gt_res=[]
        # beam_size=5
        for i in range(len(gen_candidates)):
            # randomly pick the candidate
            # len(gen_candidates)==batch_size
            lucky_boy_id=random.randint(0, len(gen_candidates[i])-1)
            lucky_boy=gen_candidates[i][lucky_boy_id]["tokens"]
            # save token
            gen_target.append(lucky_boy.int())
            # save decoded gen and ground truth
            gen_res.append(self.tokenizer.decode(lucky_boy).strip())
            # remove padding index
            ground_truth=inputs["target"][i]
            ground_truth=ground_truth[ground_truth.ne(self.padding_idx)]

            gt_res.append(self.tokenizer.decode(ground_truth))

        # gen_target: List[Tensor]
        # gen_res: List[str]
        # gt_res: List[str]
        return gen_target, gen_res, gt_res