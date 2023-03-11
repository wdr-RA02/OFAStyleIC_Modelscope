import os
import re
from modelscope.msdatasets import MsDataset

def generate_msdataset(ds_path: str, 
                       json_name: str,
                       remap_dict=None):
    '''
    load pcaptions datasets from the given json
    
    args: ds_path, json_name: 数据集位置
    remap_dict(可选): 是否需要进行remap

    return: ds hf类型的数据集
    '''
    ds=MsDataset.load('json',
                      data_files={"train":os.path.join(ds_path, json_name)}, 
                      split="train")
    if remap_dict is not None:
        # ds为hf格式的数据集
        ds=ds.remap_columns(remap_dict)
        return ds

    return ds.to_hf_dataset()

def collate_pcaption_dataset(data, dataset_dir: str, file_attr: str=".jpg"):
    '''
    将["image_hash", "text", style_key]的原始数据集转换为["image", "text", style_key]的新数据集
    原则是尽量少在preprocessor那里加参数

    args: data 原始数据集, dict
    dataset_dir: 数据集地址
    file_attr: 以点开头的后缀

    return: 新数据集
    '''

    image_path=os.path.join(dataset_dir,data["image"]+file_attr)
    if re.findall(re.compile("^(http|https|ftp|smb)://.*$"),image_path)==[]:  
        # 对网页路径采取不同的处理方式 
        image_path.replace("\\","/")
    data["image"]=image_path

    return data
