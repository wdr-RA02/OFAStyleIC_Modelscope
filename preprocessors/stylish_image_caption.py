# import torch
from modelscope.preprocessors.multi_modal import OfaPreprocessor as OfaPre
from modelscope.preprocessors.ofa import OfaImageCaptioningPreprocessor as OfaICP
from modelscope.utils.constant import ModeKeys
from typing import Any, Dict, List, Union
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import (Fields, Invoke, ModeKeys, ModelFile,
                                       Tasks)

# 按照modelscope的要求注册preprocessor
@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name="ofa-stylish-caption-preprocessor"
)                                       
class OfaPreprocessorforStylishIC(OfaPre):
    def __init__(self, 
            model_dir: str, 
            mode=ModeKeys.INFERENCE, 
            style_key="style",
            *args, **kwargs):
        '''
        OfaPreprocessorforStylishIC: 在Modelscope提供的OFAPreprocessor基础上加入
        风格要素, 具体而言包括修改preprocess为OFAStylishICP, 加入prompt以及修改data结构

        args: style_key: 指定data中代表风格的key, 默认为`style`
        '''
        super().__init__(model_dir, mode, *args, **kwargs)

        # 在OFAPreprocessor的基础上修改data preprocessor, key和tokenizer
        self.preprocess = OfaStylishICPreprocessor(cfg=self.cfg, 
                    model_dir=model_dir, 
                    mode=mode, 
                    style_key=style_key)
        # 指定style标签的key
        self.STYLE_KEY = style_key
        # self.tokenizer=self.preprocess.tokenizer
        # add "style" key to self.keys
        self.keys.append(self.STYLE_KEY)

    def __call__(self, 
            input: Union[str, tuple, Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        print(self.cfg.model.get("prompt", "not defined, use default prompt"))
        # 暂时先不修改父类的call函数
        return super().__call__(input, *args, **kwargs)
    
class OfaStylishICPreprocessor(OfaICP):
    '''
    OfaStylishICPreprocessor: 在Modelscope提供的OFAICPreprocessor基础上加入
    风格要素, 具体而言包括修改prompt以及修改dataset结构
    '''
    def __init__(self,
                cfg, 
                model_dir,
                style_key="style",
                mode=ModeKeys.INFERENCE, 
                *args, 
                **kwargs):
        super().__init__(cfg, model_dir, mode, *args, **kwargs)
        self.style_dict = dict()
        self.STYLE_KEY = style_key

    def add_style_token(self, style_list: List):
        '''
        需要的时候调用此函f "style_captions的风格逐一添加到self.tokenizer里面
        '''
        raise NotImplementedError("尚未考虑添加style token的问题")   

    def __call__(self,
                data: Dict[str, Any]) -> Dict[str, Any]:

        assert self.STYLE_KEY in data
        return super().__call__(data)

    def _build_infer_sample(
                self, 
                data: Dict[str, Any]) -> Dict[str, Any]:
        r'''
        args: data 以字典形式输入的数据, 最少包含 `image`, `prompt`, `label`
        三个主键, 风格暂时以"style"主键插入

        return 与父类的该函数一样返回包字典,包含source, image, mask and label data.
        '''
        sample: Dict[str, Any]=super()._build_infer_sample(data)
        # define the new prompt
        new_prompt=self.cfg.model.get("prompt", " what does the image describe? write a {} reply.")
        inputs=new_prompt.format(data[self.STYLE_KEY])
        # update the dict with our new prompt
        sample["source"]=self.tokenize_text(inputs)
        return sample

