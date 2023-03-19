import os.path as osp
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from modelscope.metrics.base import Metric

from modelscope.metrics.text_generation_metric import TextGenerationMetric as TxtGenMetric
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
        def pop_empty(inputs: List[str]):
            new_list=list()
            for one_str in inputs:
                if len(one_str.replace(" ","").replace(".",""))>0:
                    # for each caption C: C->{"caption":c}
                    new_list.append({"caption":one_str.lower()})
            return new_list
        
        # use filename as image_id
        image_ids=[osp.split(k[self.image_text])[-1] for k in inputs[self.sample_text]]
        # ref:{id: [{"caption":cap}]}
        self.reference.update({i:caps for i,caps in zip(image_ids,map(pop_empty, outputs[self.pred_text])) \
                       if len(caps)>0})

        # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}
        self.ground_truth.update({i:caps for i,caps in zip(image_ids,map(pop_empty, inputs[self.target_text])) \
                       if len(caps)>0})

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
        

class ImageCaptionMetric_txtgen(TxtGenMetric):
    '''
    (旧版, 已被抛弃)
    包含BLEU-1, BLEU-4, Rouge的image cap metric
    '''
    def __init__(self, 
                 target_text='labels', 
                 pred_text='caption'):
        super().__init__(target_text, pred_text)


    def collate_io_data(self, 
                        outputs: Dict[str, List[str]], 
                        inputs: Dict[str, List[str]]):
        # squeeze each ele of output["caption"]
        cap_outputs={
            self.pred_text: [k.lower() for caps in outputs[self.pred_text] for k in caps]
        }
        multi_cap_inputs={
            self.target_text: [list(map(lambda x:x.lower(), k)) for k in inputs[self.target_text]]
        }

        return cap_outputs, multi_cap_inputs


    def add(self, 
            outputs: Dict[str, List[str]], 
            inputs: Dict[str, List[str]]):
        # get the data with only captions
        outputs, mul_cap_inputs=self.collate_io_data(outputs, inputs)

        # use the first caption as ground truth in txtgen metric
        single_cap_inputs={
            self.target_text: [caps[0] for caps in mul_cap_inputs[self.target_text]]
        }
        print("Input: {}".format(single_cap_inputs))
        '''
        after
        input:{'nsentences': n,
               'net_input': {'input_ids': prompt: List[int], 
                             'patch_images': imgs: List[Tensor]...}, 
               'labels': one_data["text"]: List[List[str]]
              }
        '''
        print("Outputs: {}".format(outputs))
        '''
        after
        output:{'caption': gen_captions: List[str]}    #每个List[str]里只有一个内容    
        '''
        return super().add(outputs, single_cap_inputs)