import torch
from modelscope.preprocessors.ofa import OfaImageCaptioningPreprocessor as OfaICP
from modelscope.utils.constant import ModeKeys
from typing import Any, Dict

class OfaStylishICProcessor(OfaICP):
    '''
    OfaStylishICPreprocessing: 在Modelscope提供的OFAPreprocessor基础上加入
    风格要素, 具体而言包括修改prompt以及修改dataset结构
    '''
    def __init__(self,
                cfg, 
                model_dir, 
                mode=ModeKeys.INFERENCE, 
                *args, 
                **kwargs):      
        super().__init__(cfg, model_dir, mode, *args, **kwargs)

    def __call__(self,
                data: Dict[str, Any]) -> Dict[str, Any]:
        if "personality" not in data:
            raise KeyError("输入的数据缺少personality标签")
        return super().__call__(data)

    def _build_infer_sample(
                self, 
                data: Dict[str, Any]) -> Dict[str, Any]:
        r'''
        args: data 以字典形式输入的数据, 最少包含 `image`, `prompt`, `label`
        三个主键, 风格暂定以"personality"主键插入

        return 与父类的该函数一样返回包字典,包含source, image, mask and label data.
        '''
        sample: Dict[str, Any]=super()._build_infer_sample(data)
        # define the new prompt
        new_prompt=self.cfg.model.get("prompt", " write a '{}' description of the image")
        # inputs=new_prompt.format(data["personality"])
        # update the dict with our new prompt
        sample.update("source", self.tokenize_text(inputs))
        return sample
