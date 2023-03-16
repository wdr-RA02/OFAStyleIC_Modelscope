import os
import re
from typing import Dict
from datasets.formatting.formatting import LazyRow
from modelscope.msdatasets import MsDataset

def generate_msdataset(ds_path: str,
                            json_name: str,
                            remap_dict=None):
    '''
    load pcaptions datasets from the given json, supports import of test_set
    
    args: ds_path, json_name: 数据集位置
    remap_dict(可选): 是否需要进行remap

    return: ds hf类型的数据集
    '''
    ds=MsDataset.load('json',
                      data_files={"train":os.path.join(ds_path, json_name)}, 
                      split="train").to_hf_dataset()
    # 将additional_comments合并到comment里面
    if "additional_comments" in ds.column_names:
        def change(x):
            # print(type(x))
            assert isinstance(x, LazyRow)
            x["comment"]=[x["comment"], *x["additional_comments"]]
            return x
        
        ds=ds.map(change, remove_columns="additional_comments")
        
    if remap_dict is not None:
        assert isinstance(remap_dict, dict)
        # ds为hf格式的数据集
        ds=ds.rename_columns(remap_dict)

    return ds

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
