from .constants import *

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

def list_styles(dataset_path: str,
                style_file_name: str):
    '''
    从personality_captions中加载风格列表
    
    args: style_file_name: personality.txt

    return: style_list
    '''

    style_list_dir=os.path.join(dataset_path, style_file_name)
    with open(style_list_dir) as f:
        style_list=f.read().strip("\n").split("\n")
    
    return style_list

def get_style_dict_from_ls(style_list: list):
    '''
    需要的时候调用此函数将personality_captions的风格逐一添加到self.tokenizer里面
    '''
    # 暂时没想到怎么加style_k, 先用code_k顶一下吧, 哎...
    style_dict = {style:f"<code_{i}>" for i, style in enumerate(style_list)}

    return style_dict

def generate_style_dict(train_conf: dict):
    '''
    根据train_conf直接返回需要的style_dict
    '''
    style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
    return get_style_dict_from_ls(style_list)
