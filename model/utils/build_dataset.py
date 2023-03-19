from functools import partial
import os
import re
from datasets.formatting.formatting import LazyRow
from modelscope.msdatasets import MsDataset

def generate_msdataset(ds_path: str,
                       json_name: str,
                       remap_dict=None,
                       parts: str=""):
    '''
    load pcaptions datasets from the given json, supports import of test_set
    
    args: 
    ds_path, json_name:     数据集位置
    remap_dict(可选):       是否需要进行remap
    parts(可选): str"[a:b]" 是否切片数据集
    
    return: ds hf类型的数据集
    '''
    ds=MsDataset.load('json',
                      data_files={"train":os.path.join(ds_path, json_name)}, 
                      split="train"+parts).to_hf_dataset()
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

def generate_train_eval_ds(train_conf: dict,
                           train: bool=False,
                           eval: bool=False,
                           remap: dict=None):
    '''
    生成hf格式的数据集

    args: train_conf: 训练配置文件
    remap: 重映射dict

    return: train_ds, eval_ds
    '''
    img_addr=train_conf["img_addr"]
    dataset_path=train_conf["dataset_path"]
    collate_fn=partial(collate_pcaption_dataset, dataset_dir=img_addr)
    train_ds, eval_ds=None, None
    # generate dataset according to needs
    if train:
        train_ds = generate_msdataset(dataset_path,
                                    train_conf["train_json"],
                                    remap)
        train_ds = train_ds.map(collate_fn)
    if eval:
        eval_ds = generate_msdataset(dataset_path, 
                                    train_conf["test_json"],
                                    remap)

        eval_ds = eval_ds.map(collate_fn)

    return train_ds, eval_ds

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
