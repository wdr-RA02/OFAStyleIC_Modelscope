from torchvision import transforms
from datasets.formatting.formatting import LazyDict
# import torch
from modelscope.preprocessors.multi_modal import OfaPreprocessor as OfaPre
from modelscope.preprocessors.ofa import OfaImageCaptioningPreprocessor as OfaICP
from modelscope.utils.constant import ModeKeys
from typing import Any, Callable, Dict, List, Union
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import (Fields, ModeKeys)

# 按照modelscope的要求注册preprocessor
@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name="ofa-stylish-ic-preprocessor")                                       
class OfaPreprocessorforStylishIC(OfaPre):
    def __init__(self, 
            model_dir: str, 
            mode=ModeKeys.INFERENCE, 
            cfg_modify_fn: Callable=None,
            *args, 
            **kwargs):
        '''
        OfaPreprocessorforStylishIC: 在Modelscope提供的OFAPreprocessor基础上加入
        风格要素, 具体而言包括修改preprocess为OFAStylishICP, 加入prompt以及修改data结构

        args: 
        model_dir: str: 模型位置
        mode: ("train", "eval", "inference"): preprocessor状态
        cfg_modify_fn: cfg修改函数
        '''
        super().__init__(model_dir, mode, *args, **kwargs)
        self.cfg_modify_fn=cfg_modify_fn
        if self.cfg_modify_fn is not None:
            # 在trainer就位之前先通过cfg_modify_fn修改好cfg
            self.cfg=self.cfg_modify_fn(self.cfg)
        # 在OFAPreprocessor的基础上修改data preprocessor, key和tokenizer
        self.preprocess = OfaStylishICPreprocessor(cfg=self.cfg, 
                    model_dir=model_dir, 
                    mode=mode)
        # 指定style标签的key
        self.STYLE_KEY = "style"
        self.tokenize_style=self.preprocess.tokenize_style
        self.tokenizer=self.preprocess.tokenizer
        # add "style" key to self.keys
        if not self.STYLE_KEY in self.keys:
            self.keys.append(self.STYLE_KEY)
        print(f"OFAPpSIC registered, model_dir:{model_dir}")


    def __call__(self, 
            input: Union[str, tuple, Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        # print(self.cfg.model.get("prompt", "not defined, use default prompt"))
        # 对于hf datasets的map函数, 要特别处理一下input
        if isinstance(input, LazyDict):
            input=dict(input)
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
                mode=ModeKeys.INFERENCE, 
                *args, 
                **kwargs):
        '''
        初始化Preprocessor

        args: style_key: dataset中存储风格的key
        dataset_dir: 指定数据集目录
        dataset_file_attr: 指定图片后缀(必须带点)
        '''
        super().__init__(cfg, model_dir, mode, *args, **kwargs)
        # add random crop to 224x224 to match other works
        assert self.patch_image_size<=self.max_image_size
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize(
                (self.max_image_size, self.max_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(
                (self.patch_image_size, self.patch_image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        # style tokenizer
        self.style_dict = None
        self.tokenize_style = False

        self.STYLE_KEY="style"

    def __call__(self,
                data: Dict[str, Any]) -> Dict[str, Any]:
        
        assert self.STYLE_KEY in data
        return super().__call__(data)
    
    def add_style_token(self, style_dict: Dict[str, str]):
        self.style_dict = style_dict
        # Format: {style: "<code_k>"}
        print("Got style dict, len={}".format(len(style_dict)))
        self.tokenizer.add_tokens(list(self.style_dict.values()))
        # open the token mode
        self.tokenize_style = isinstance(self.style_dict, dict)

    def _build_infer_sample(
                self, 
                data: Dict[str, Any]) -> Dict[str, Any]:
        r'''
        args: data 以字典形式输入的数据, 最少包含 `image`, `prompt`, `label`
        三个主键, 风格暂时以"style"主键插入

        如果只输入image_hash, 那么必须通过set_dataset_dir指定数据集目录和文件后缀
        return 与父类的该函数一样返回字典,包含source, image, mask and label data.
        '''

        sample: Dict[str, Any]=super()._build_infer_sample(data)
        # define the new prompt
        new_prompt=self.cfg.model.get("prompt", " what does the image describe? write a {} reply.")

        # get current style
        cur_style=self.style_dict.get(data[self.STYLE_KEY], "<unk>") if self.tokenize_style \
                  else data[self.STYLE_KEY]
        # 教训惨痛, 遂决定添加warning
        if cur_style=="<unk>":
            print("WARNING: Got unknown style token, check orig: {}".format(data[self.STYLE_KEY]))
        inputs=new_prompt.format(cur_style)
        # update the dict with our new prompt
        sample["source"]=self.tokenize_text(inputs)
        return sample
    

