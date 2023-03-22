from metric.utils import pop_empty
from typing import List
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from string import punctuation

class RewardCalculator(object):
    def __init__(self):
        # reference=sample_sequence
        self.PTB=PTBTokenizer()
        self.cider=Cider()    

    def set_ref_and_gts(self, 
                        reference: List[str], 
                        ground_truth: List[str]):
        '''
        load reference (sampled or greedy_decoded) and ground_truth (inputs["labels"]) here
        '''
        self.reference=reference
        # ground_truth is just ground truth! 
        self.ground_truth=ground_truth
        assert len(reference)==len(ground_truth), \
            "size of ref list and gt list must be identical"
        self.batch_size=len(reference)
        
        # ref:{id: [{"caption":cap}]}
        # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}

        
    def __call__(self, 
                 reference:List[str], 
                 ground_truth:List[str]) -> List[float]:
        
        self.set_ref_and_gts(reference, ground_truth)
        return self.calc_cider()

    def convert_ref(self, ref, gts):
        '''
        convert seq to the required format
        '''
        ref_out={"img_{}".format(i):pop_empty(cap)\
                  for i, cap in enumerate(ref)}
        
        gts_out={"img_{}".format(i):pop_empty(cap)\
            for i, cap in enumerate(gts)}
        
        return ref_out, gts_out
    
    def convert_gts(self, gts):
        '''
        gts=inputs["labels"]
        '''
        pass
    
    def calc_cider(self):
        ref_batch, gts_batch=self.convert_ref(self.reference, self.ground_truth)
        '''
        after
        gth={id:[caption_1]}
        '''

        # expand the list to outside, ie: gth=>[{id: caption}]
        ref_batch=[{id: [cap]} for ref in self.PTB(ref_batch) \
                   for id,cap in ref.items()]
        gts_batch=[{id: [cap]} for ref in self.PTB(gts_batch) \
                   for id,cap in ref.items()]
        
        scores=[]
        for ref, gts in ref_batch, gts_batch:
            # cal score separately for each pair in batch
            score, _=self.cider.compute_score(gts, ref)
            scores.append(score.round(3))
        
        return scores
