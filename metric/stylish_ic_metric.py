from modelscope.metrics.builder import METRICS
from modelscope.utils.registry import default_group

from modelscope.metrics.text_generation_metric import TextGenerationMetric as TxtGenMetric
from typing import List, Dict

@METRICS.register_module(
    group_key=default_group, module_name="image-caption-metric")
class ImageCaptionMetric(TxtGenMetric):
    '''
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
        
        outputs, inputs=self.collate_io_data(outputs, inputs)

        # use the first caption as ground truth in txtgen metric
        single_cap_inputs={
            self.target_text: [caps[0] for caps in inputs[self.target_text]]
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
                'samples': inputs: Dict[str, Union[str, list]]
        
        '''
        return super().add(outputs, single_cap_inputs)

    def evaluate(self):
        return super().evaluate()