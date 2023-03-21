from .utils import *
from modelscope.metrics.base import Metric
from typing import List, Dict

class ImageCaptionMetric(Metric):
    '''
    a custom ICMetric implemented using pycocoeval

    args:
    pred_text: eval数据集中含有caption的key名字
    target_text: 模型预测输出中含有caption的key名字
    mul_100: 得到的分数是否乘以100
    '''
    def __init__(self, 
                 pred_text: str="caption",
                 target_text: str="labels",
                 sample_text: str="samples",
                 image_text: str="image",
                 mul_100: bool=False
                 ):
        print("Using pycocoeval metric currently. ")
        self.pred_text=pred_text
        self.target_text=target_text
        self.sample_text=sample_text
        self.image_text=image_text
        self.multiply=(1.,100.)[int(mul_100)]
        # reference and ground truth dicts
        self.reference=dict()
        self.ground_truth=dict()
        # init metrics
        self.eval_metrics=[
            (["BLEU_1", "BLEU_4"], Bleu(4)),
            ("ROUGE_L", Rouge()),
            ("CIDEr", Cider()),
            ("SPICE", Spice())
        ]
        self.tokenizer=PTBTokenizer()

    def add(self, outputs: Dict, inputs: Dict):
        # squeeze each ele of output["caption"]
        # image tensors, no use at all
        inputs.pop("net_input",None)
        dicts=convert_from_dataset(outputs, inputs,
                                   pred_text=self.pred_text,
                                   target_text=self.target_text,
                                   sample_text=self.sample_text,
                                   image_text=self.image_text)
        # ref:{id: [{"caption":cap}]}
        self.reference.update(dicts[0])

        # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}
        self.ground_truth.update(dicts[1])

        # print("Input: {}".format(self.ground_truth))
        # print("Outputs: {}".format(self.reference))
        
    
    def _get_bleu_score_dict(self, 
                             label: list, 
                             score: list):
        '''
        return a flat BLEU score dict with given label

        args:
        label: BLEU item want to be included, **must be ["BLEU_k"] format**
        score: origin score list with all four BLEU scores
        return: out_dict: a dict {"BLEU_k": score[k-1]}
        '''
        out_dict=dict()
        complete_bleu=["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]
        for i,item in enumerate(complete_bleu):
            if item in label:
                out_dict.update({item: score[i]*self.multiply})
        
        return out_dict
            
    def evaluate(self):
        '''
        after
        ref:{id:[caption]} 
        '''
        # tokenize the content using PTBTokenizer first
        reference=self.tokenizer.tokenize(self.reference)
        '''
        after
        gth={id:[caption_1, caption_2, ...]}
        '''
        ground_truth=self.tokenizer.tokenize(self.ground_truth)
        eval_results=dict()
        for label, metric in self.eval_metrics:
            print("Evaluating {}...".format(label))
            score, _=metric.compute_score(ground_truth, reference)
            # form the dict
            if isinstance(score, list):
                result=self._get_bleu_score_dict(label, score)
            else:
                result={label:score*self.multiply}
            eval_results.update(result)
        return eval_results


    def merge(self, other: 'ImageCaptionMetric'):
        self.reference.update(other.reference)
        self.ground_truth.update(other.ground_truth)
        