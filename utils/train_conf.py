import os
import json


def load_train_conf(train_conf_filename: str):
    '''
    加载训练文件配置json

    args: train_conf_filename: 配置文件名称, 默认为trainer_config.json

    return: json化后的配置文件字典
    '''
    # 若加载失败则返回none
    train_conf = None
    with open(train_conf_filename) as f:
        train_conf=json.load(f)
    return train_conf

def list_styles(style_file_name: str):
    '''
    从personality_captions中加载风格列表
    
    args: style_file_name: personality.txt

    return: style_list
    '''
    train_conf=load_train_conf("trainer_config.json")
    assert isinstance(train_conf, dict)

    style_list_dir=os.path.join(train_conf["dataset_path"], style_file_name)
    with open(style_list_dir) as f:
        style_list=f.read().strip("\n").split("\n")
    
    return style_list

def add_style_token(style_list: list):
    '''
    需要的时候调用此函数将personality_captions的风格逐一添加到self.tokenizer里面
    '''
    style_dict = {style:i for i, style in enumerate(style_list)}
    # raise NotImplementedError("尚未考虑添加style token的问题")

    return style_dict