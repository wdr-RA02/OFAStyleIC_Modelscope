from typing import Any, Callable, Dict, Union

from datasets.formatting.formatting import LazyDict
from modelscope.preprocessors.builder import PREPROCESSORS
# import torch
from modelscope.preprocessors.multi_modal import OfaPreprocessor as OfaPre
from modelscope.preprocessors.ofa import \
    OfaImageCaptioningPreprocessor as OfaICP

from modelscope.preprocessors.ofa.utils.collate import collate_fn
from modelscope.utils.constant import Fields, ModeKeys
from torchvision import transforms
from torch import tensor


# 按照modelscope的要求注册preprocessor
@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name="ofa-stylish-ic-preprocessor")                                       
class OfaPreprocessorforStylishIC(OfaPre):
    def __init__(self, 
            model_dir: str, 
            mode=ModeKeys.INFERENCE, 
            cfg_modify_fn: Callable=None,
            cider=False,
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
        self.cider=cider
        self.preprocess = OfaStylishICPreprocessor(cfg=self.cfg, 
                    model_dir=model_dir, 
                    mode=mode,
                    cider=self.cider)
        # 指定style标签的key
        self.STYLE_KEY = "style"
        self.tokenize_style=self.preprocess.tokenize_style
        self.tokenizer=self.preprocess.tokenizer
        # add "style" key to self.keys
        if not self.STYLE_KEY in self.keys:
            self.keys.append(self.STYLE_KEY)
        # different training steps require different method
        print(f"OFAPpSIC registered, model_dir:{model_dir}")
        print("CIDEr finetune: {}".format(self.cider))


    def __call__(self, 
            input: Union[str, tuple, Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        # print(self.cfg.model.get("prompt", "not defined, use default prompt"))
        # 对于hf datasets的map函数, 要特别处理一下input
        if isinstance(input, LazyDict):
            input=dict(input)
        # 因为添加了ITM任务, 所以需要把基类的call函数照抄过来修改一下
        # 主要是把[sample]那里改掉
        # return super().__call__(input, *args, **kwargs)
        if isinstance(input, dict):
            data = input
        else:
            data = self._build_dict(input)
        sample = self.preprocess(data)
        # sample=[(CAP_0,ITM_0), (CAP_1,ITM_1), ...]
        str_data = dict()
        for k, v in data.items():
            str_data[k] = str(v)
        # print(sample, type(sample))
        for item in sample:
            item['sample'] = str_data
        if self.no_collate:
            return sample
        else:
            return collate_fn([item for item in sample],
                              pad_idx=self.tokenizer.pad_token_id,
                              eos_idx=self.tokenizer.eos_token_id)
    
class OfaStylishICPreprocessor(OfaICP):
    '''
    OfaStylishICPreprocessor: 在Modelscope提供的OFAICPreprocessor基础上加入
    风格要素, 具体而言包括修改prompt以及修改dataset结构
    '''
    def __init__(self,
                cfg, 
                model_dir,
                mode=ModeKeys.INFERENCE,
                cider=False, 
                *args, 
                **kwargs):
        '''
        初始化Preprocessor

        args: style_key: dataset中存储风格的key
        dataset_dir: 指定数据集目录
        dataset_file_attr: 指定图片后缀(必须带点)
        '''
        super().__init__(cfg, model_dir, mode, *args, **kwargs)
        print("max_image_size={}".format(self.cfg.model.max_image_size))
        print("patch_image_size={}".format(self.cfg.model.patch_image_size))
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
        self.cider=cider

        self.STYLE_KEY="style"

    def __call__(self,
                data: Dict[str, Any]) -> Dict[str, Any]:
        
        assert self.STYLE_KEY in data
        # since data_collate fn needs changing, all output need to be tuple
        sample=super().__call__(data)
        if not isinstance(sample, tuple):
            sample=(sample, )
        return sample
    
    def add_style_token(self, style_dict: Dict[str, str]):
        self.style_dict = style_dict
        # Format: {style: "<code_k>"}
        print("Got style dict, len={}".format(len(style_dict)))
        self.tokenizer.add_tokens(list(self.style_dict.values()))
        # open the token mode
        self.tokenize_style = isinstance(self.style_dict, dict)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # caption task
        sample_caption=super()._build_train_sample(data)
        # add weight
        sample_caption["conf"]=tensor([1.0])

        #
        sample_itm=None
        # output sample
        if self.cider:
            sample=(sample_caption, )
        else:
            sample=(sample_caption, )

        return sample

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
        # for unknown style, we use <code_i+1> instead of <unk>
        cur_style=self.style_dict.get(data[self.STYLE_KEY], "<code_{}>".format(len(self.style_dict))) if self.tokenize_style \
                  else data[self.STYLE_KEY]
        # 教训惨痛, 遂决定添加warning
        if cur_style=="<code_{}>".format(len(self.style_dict)):
            print("WARNING: Got unknown style token, check orig: {}".format(data[self.STYLE_KEY]))
        inputs=new_prompt.format(cur_style)
        # update the dict with our new prompt
        sample["source"]=self.tokenize_text(inputs)
        # weight

        return sample
    

