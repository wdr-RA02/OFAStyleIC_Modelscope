from metric.utils import pop_empty
from typing import List
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class RewardCalculator(object):
    def __init__(self, eos_token: str="</s>", **kwargs):
        # reference=sample_sequence   
        self.eos_token = eos_token
        self.PTB=PTBTokenizer()
        self.cider=Cider() 

    def set_ref_and_gts(self, 
                        reference: List[str], 
                        ground_truth: List[str],
                        eos_token: str=None):
        '''
        load reference (sampled or greedy_decoded) and ground_truth (inputs["labels"]) here
        '''
        self.reference=reference
        # ground_truth is just ground truth! 
        self.ground_truth=ground_truth
        assert len(reference)==len(ground_truth), \
            "size of ref list and gt list must be identical"
        self.batch_size=len(reference)

        if isinstance(eos_token, str):
            self.eos_token=eos_token
        
        # ref:{id: [{"caption":cap}]}
        # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}

        
    def __call__(self, 
                 reference:List[str], 
                 ground_truth:List[str]) -> List[float]:
        
        self.set_ref_and_gts(reference, ground_truth)
        cider=self.calc_cider()

        return cider

    def convert(self, ref, gts):
        '''
        convert seq to the required format
        '''
        ref_out={"img_{}".format(i):pop_empty([cap])\
                  for i, cap in enumerate(ref)}
        
        gts_out={"img_{}".format(i):pop_empty([cap])\
            for i, cap in enumerate(gts)}
        
        return ref_out, gts_out
    
    
    def calc_cider(self):
        ref_batch, gts_batch=self.convert(self.reference, self.ground_truth)
        '''
        after
        gth={id:[caption_1]}
        '''
        ref_batch=self.PTB.tokenize(ref_batch)
        gts_batch=self.PTB.tokenize(gts_batch)
        # don't do this, or CIDEr will be 0 forever
        # # expand the list to outside, ie: gth=>[{id: caption}]
        # ref_batch=[{id: ref_batch[id]} for id in ref_batch]
        # gts_batch=[{id: gts_batch[id]} for id in gts_batch]
        
        # for ref, gts in zip(ref_batch, gts_batch):
        #     # cal score separately for each pair in batch
            
        #     scores.append(score)
        score, _=self.cider.compute_score(gts_batch, ref_batch)

        return score*100
