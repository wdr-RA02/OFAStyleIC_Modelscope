from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from metric.utils import pop_empty
from typing import List, Union, Dict, Any
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

class RewardCalculator(object):
    def __init__(self, 
                 eos_token: str="</s>",
                 mul_100:bool=False, 
                 **kwargs):
        '''
        基于pycocoevalcap实现的scst reward计算
        args: 
        eos_token
        mul_100: 得到的cider分数是否乘以100

        调用return:
        score: batch的平均cider
        scores: list, batch中每一个描述的cider
        '''
        # reference=sample_sequence
        self.eos_token = eos_token
        self.PTB=PTBTokenizer
        self.cider=Cider()
        self.multiply_constant=(1,100)[mul_100]

    OK_LIST_FMT=List[Union[str, List[str]]]   

    def get_tokenized_sequence(self,
                               sequence_lst: OK_LIST_FMT):
        
        seq_list=self.convert(sequence_lst)    
        # ground truth only need to be tokenized once
        seq_batch=self.PTB().tokenize(seq_list)

        return seq_batch
    
    def set_ref_and_gts(self, 
                        reference: OK_LIST_FMT, 
                        ground_truth: OK_LIST_FMT):
        '''
        load reference (sampled or greedy_decoded) and ground_truth (inputs["labels"]) here

        self.ref/self.gts is the tokenized sequence
        '''
        thPool=ThreadPoolExecutor(max_workers=2)
        results=[]
        for seq in reference, ground_truth:
            results.append(thPool.submit(self.get_tokenized_sequence, seq))
        
        # get results from thread pool
        self.reference=results[0].result()
        self.ground_truth=results[1].result()

        # ground_truth is just ground truth! 
        self.batch_size=len(reference)
        
       
    def convert(self, ref):
        '''
        convert seq to the required format
        '''
        ref_out={"img_{}".format(i):pop_empty(cap if isinstance(cap,list) else [cap])\
                  for i, cap in enumerate(ref)}
        # ref:{id: [{"caption":cap}]}
        return ref_out
    
    
    def calc_absolute_reward(self):
        '''
        make sure self.reference and self.ground_truth are both tokenized before invoking the fn!
        '''
        # don't do this, or CIDEr will be 0 forever
       
        # for ref, gts in zip(ref_batch, gts_batch):
        #     # cal score separately for each pair in batch
            
        #     scores.append(score)
        score, scores=self.cider.compute_score(self.ground_truth, self.reference)

        return score*self.multiply_constant, scores*self.multiply_constant


    def __call__(self, 
                 reference:Union[Dict[str, OK_LIST_FMT], OK_LIST_FMT], 
                 ground_truth:OK_LIST_FMT,
                 eos_token: str=None) -> List[float]:
        '''
        基于pycocoevalcap实现的scst reward计算
        args: 
        reference: 可从下面二选一: 
        - sample得到的序列或greedy decode的序列, 格式为List[str]
        - 将两个序列作为一个字典一起加进来, 格式为``{"sampled": List[str], "greedy": List[str]}``
        ground_truth: 从数据集中得到的caption
        eos_token: tokenizer使用的eos_token

        调用return:
        score: batch的平均cider
        scores: list, batch中每一个描述的cider
        '''
        if isinstance(eos_token, str):
            self.eos_token=eos_token
        if isinstance(reference, dict):
            assert set(reference.keys()) == {"sampled", "greedy"}
            ref_sampled, ref_baseline=reference["sampled"], reference["greedy"]

            assert len(ref_sampled)==len(ground_truth)==len(ref_baseline), \
                "size of ref lists and gt list must be identical"
            self.batch_size = len(ref_sampled)
            # rewards score
            rewards, reward_lists=list(), list()
            thPool=ThreadPoolExecutor(max_workers=2)
            results=[]
            # tokenized gts for only once          
            self.ground_truth=self.get_tokenized_sequence(ground_truth)

            for reference in ref_sampled, ref_baseline:
                results.append(thPool.submit(self.get_tokenized_sequence, reference))
            ref_sampled=results[0].result()
            ref_baseline=results[1].result()

            for reference in ref_sampled, ref_baseline:
                self.reference=reference
                score, scores_=self.calc_absolute_reward()
                rewards.append(score)
                reward_lists.append(scores_)
            
            # return the subtraction of the two rewards
            cider=(rewards[0]-rewards[1], reward_lists[0]-reward_lists[1])

        elif isinstance(reference, list):
            assert len(reference)==len(ground_truth), \
                "size of ref list and gt list must be identical"
            
            self.set_ref_and_gts(reference, ground_truth)
            cider=self.calc_absolute_reward()
        else:
            raise TypeError("reference expected dict or list, got {}".format(type(reference)))

        return cider
