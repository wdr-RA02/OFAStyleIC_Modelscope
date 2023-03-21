import random
import numpy as np
import torch
import torch.nn.functional as F
from modelscope.models.multi_modal import OfaForAllTasks
from torch.nn.modules.loss import _Loss


class SelfCriticalSeqTrainingCriterion(_Loss):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, model, inputs):
        pass

    def get_sample_from_beams(self, model, inputs):
        '''
        从model的inference输出(beam_size个)随机sample一个作为sampled sequence

        args:
        model: OfaModelForAll
        inputs: train preprocessor的输出
        
        return:
        gen_target: decode后的sampled_sequence
        gen_res: sampled_sequence的token
        '''

        assert isinstance(model, OfaForAllTasks), \
            "model required to be OfaForAllTasks type, got {}".format(type(model))
        
        model.model.eval()
        with torch.no_grad():
            # generate the candidates with beam decoding
            gen_candidates=model.generator.generate([model.model],
                                              inputs,
                                              prefix_tokens=input.get(
                                                  'prefix_tokens', None))
        gen_target=[]
        gen_res=[]
        gt_res=[]
        for i in len(gen_candidates):
            # randomly pick the candidate
            # 虽然很确信i一定只能是0, 但是还是先这么写着吧
            lucky_boy=random.choice(gen_candidates[i])
            # save token
            gen_target.append(lucky_boy.int())
            # save decoded gen and ground truth
            gen_res.append(model.tokenizer.batch_decode(lucky_boy))

            gt_res.append(model.tokenizer.batch_decode(inputs["target"]))
        return gen_target, gen_res, gt_res