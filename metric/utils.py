import os
from typing import List
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def convert_from_dataset(outputs,
                         inputs,
                         pred_text: str="caption",
                         target_text: str="labels",
                         sample_text: str="samples",
                         image_text: str="image"):
    '''
    将dataset的输入转换为pycocoevalcap需要的输入格式
    具体为: 
    ref={id: [{"caption":cap}]}
    gt={id: [{"caption":cap_1}, {"caption":cap_2}...]}

    args: 
    outputs: 模型的输出, 包含参考caption
    inputs: 模型的输入, 包含ground truth
    *_text: inputs中各字段的名称

    return: reference, ground_truth
    '''
    def pop_empty(inputs: List[str]):
        new_list=list()
        for one_str in inputs:
            if len(one_str.replace(" ","").replace(".",""))>0:
                # for each caption C: C->{"caption":c}
                new_list.append({"caption":one_str.lower()})
        return new_list

    # use filename as image_id
    image_ids=[os.path.split(k[image_text])[-1] for k in inputs[sample_text]]
    # ref:{id: [{"caption":cap}]}
    reference={i:caps for i,caps in zip(image_ids,map(pop_empty, outputs[pred_text])) \
                    if len(caps)>0}

    # ground_truth={id: [{"caption":cap_1}, {"caption":cap_2}...]}
    ground_truth={i:caps for i,caps in zip(image_ids,map(pop_empty, inputs[target_text])) \
                    if len(caps)>0}
    
    return reference, ground_truth