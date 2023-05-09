import random

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
from torch import tensor, cat


# 按照modelscope的要求注册preprocessor
@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name="ofa-stylish-ic-preprocessor")                                       
class OfaPreprocessorforStylishIC(OfaPre):
    def __init__(self, 
            model_dir: str, 
            mode=ModeKeys.INFERENCE, 
            cfg_modify_fn: Callable=None,
            use_itm=False,
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
        self.itm=use_itm
        self.preprocess = OfaStylishICPreprocessor(cfg=self.cfg, 
                    model_dir=model_dir, 
                    mode=mode,
                    use_itm=self.itm)
        # 指定style标签的key
        self.STYLE_KEY = "style"
        self.tokenize_style=self.preprocess.tokenize_style
        self.tokenizer=self.preprocess.tokenizer
        # add "style" key to self.keys
        if not self.STYLE_KEY in self.keys:
            self.keys.append(self.STYLE_KEY)
        # different training steps require different method
        print(f"OFAPpSIC registered, model_dir:{model_dir}")
        if mode==ModeKeys.TRAIN:
            print("ITM in task: {}".format(self.itm))


    def __call__(self, 
            input: Union[str, tuple, Dict[str, Any]], *args, **kwargs) -> Dict[str, Any]:
        # print(self.cfg.model.get("prompt", "not defined, use default prompt"))
        # 对于hf datasets的map函数, 要特别处理一下input
        if isinstance(input, LazyDict):
            input=dict(input)
        # 因为添加了ITM任务, 所以需要把基类的call函数照抄过来修改一下
        # 主要是把[sample]那里改掉
        # return super().__call__(input, *args, **kwargs)
        if self.mode!=ModeKeys.INFERENCE:
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
        else:
            # for inference mode, just use the original call method
            return super().__call__(input)
    
class OfaStylishICPreprocessor(OfaICP):
    '''
    OfaStylishICPreprocessor: 在Modelscope提供的OFAICPreprocessor基础上加入
    风格要素, 具体而言包括修改prompt以及修改dataset结构
    '''
    def __init__(self,
                cfg, 
                model_dir,
                mode=ModeKeys.INFERENCE,
                use_itm=False, 
                style_token="<code_{}>",
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
        self.itm=use_itm
        self.style_token=style_token

        self.STYLE_KEY="style"

    def __call__(self,
                data: Dict[str, Any]) -> Dict[str, Any]:
        
        assert self.STYLE_KEY in data
        # since data_collate fn needs changing, all output need to be tuple
        sample=super().__call__(data)

        # inference mode does not require 'sample' to be tuple
        if self.mode!=ModeKeys.INFERENCE and not isinstance(sample, tuple):
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

        # randomly selects another style
        pos_style=self.style_dict.get(data[self.STYLE_KEY], self.style_token.format(len(self.style_dict))) if self.tokenize_style \
                  else data[self.STYLE_KEY]
        while True:
            # tokenize the style for comparison
            if not self.tokenize_style:
                orig_style=self.style_dict.get(data[self.STYLE_KEY], self.style_token.format(len(self.style_dict)))
            else:
                orig_style=pos_style
            other_style=random.randint(0,len(self.style_dict)-1)
            if self.style_token.format(other_style)!=orig_style:
                break
        
        # if we choose not to tokenize style, we need to check the item style
        if self.tokenize_style:
            other_style=self.style_token.format(other_style)
        else:
            other_style=list(self.style_dict.keys())[other_style]
        
        # if scst is adopted, then we should quit asap
        if self.itm:
            sample=(sample_caption, )
            return sample
        
        # itm sample
        # replies
        itm_reply=[" yes", " no", " personality"]
        itm_caption_src=sample_caption["label"]
        itm_style=pos_style

        roulette=random.random()
        if roulette<=0.5:
            # keep everything as is
            itm_target=self.tokenize_text(itm_reply[0], add_bos=False)
        elif 0.5<roulette<=0.75:
            # mess up the caption
            # TODO: random select negative captions
            itm_caption_src=random.choice(data["negative_caps"])
            # also change the index below
            itm_target=self.tokenize_text(itm_reply[1], add_bos=False)
        else:
            # mess up the style
            itm_style=other_style
            itm_target=self.tokenize_text(itm_reply[2], add_bos=False)

        itm_prev=cat([self.bos_item, itm_target[:-1]])
        # strip the caption
        itm_caption = itm_caption_src.translate(self.transtab).strip()
        cap_token_list = itm_caption.strip().split()
        itm_caption = ' '.join(cap_token_list[:self.max_tgt_length-10])
        # prompt
        itm_prompt=self.tokenize_text(' does the image describe " {} " in personality {}?'.format(itm_caption, itm_style))
        
        sample_itm={
            "patch_image": sample_caption["patch_image"],
            "patch_mask": sample_caption["patch_mask"],
            "conf": tensor([0.6]),
            "source": itm_prompt,
            "target": itm_target,
            "prev_output_tokens": itm_prev,
            "label": None
            # target :" yes/no/personality</s>"
            # prev_output_tokens
        }
        # output sample
        sample=(sample_caption, sample_itm)

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
        new_prompt=self.cfg.model.get("prompt", " what does the image describe? reply in personality {}.")
        # get current style
        # for unknown style, we use <code_i+1> instead of <unk>
        cur_style=self.style_dict.get(data[self.STYLE_KEY], self.style_token.format(len(self.style_dict))) if self.tokenize_style \
                  else data[self.STYLE_KEY]
        # 教训惨痛, 遂决定添加warning
        if cur_style==self.style_token.format(len(self.style_dict)):
            print("WARNING: Got unknown style token, check orig: {}".format(data[self.STYLE_KEY]))
        inputs=new_prompt.format(cur_style)
        # update the dict with our new prompt
        sample["source"]=self.tokenize_text(inputs)

        return sample
    

