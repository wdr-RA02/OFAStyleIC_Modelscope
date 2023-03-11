from functools import partial
from modelscope.msdatasets import MsDataset
from preprocessors.stylish_image_caption import OfaPreprocessorforStylishIC, collate_pcaption_dataset
from modelscope.utils.hub import snapshot_download
from modelscope.trainers import build_trainer
from modelscope.utils.constant import Fields, ModeKeys, ConfigKeys
from modelscope.trainers.multi_modal import OFATrainer
from transformers import default_data_collator
from modelscope.metainfo import Trainers

import os
import json

def generate_msdataset(ds_path: str, json_name: str):
    """load pcaptions datasets from the given json"""
    ds=MsDataset.load('json',
                    data_files={"train":os.path.join(ds_path, json_name)}, split="train")
    ds=ds.remap_columns({
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    })
    return ds

def cfg_modify_fn(cfg):
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'interval': 2
    }, {
        'type': 'TextLoggerHook',
        'interval': 1
    }, {
        'type': 'IterTimerHook'
    }]
    cfg.train.max_epochs=2
    return cfg

if __name__=="__main__":
    # load args from config file
    with open("trainer_config.json") as f:
        train_conf=json.load(f)
    img_addr=train_conf["img_addr"]
    dataset_path=train_conf["dataset_path"]
    train_ds = generate_msdataset(dataset_path, train_conf["train_json"])
    eval_ds = generate_msdataset(dataset_path, train_conf["val_json"])

    # 处理数据集映射
    collate_fn=partial(collate_pcaption_dataset, dataset_dir=img_addr)
    train_ds = train_ds.map(collate_fn)
    print(train_ds[0])

    model_name=train_conf["model_name"]
    # model_dir = snapshot_download(model_name)
    model_dir="workspace"
    # set dataset addr
    args = dict(
        model=model_name, 
        model_revision='v1.0.1',
        train_dataset=train_ds,
        cfg_modify_fn=cfg_modify_fn,
    )
    trainer = build_trainer(name=Trainers.ofa, default_args=args)

    # 为保安全加上这条assert
    assert type(trainer)==OFATrainer
    work_dir = args.get("work_dir", "workspace")
    preprocessor = {
        ConfigKeys.train:
            OfaPreprocessorforStylishIC(
                model_dir=work_dir,mode=ModeKeys.TRAIN, no_collate=True),
        ConfigKeys.val:
            OfaPreprocessorforStylishIC(
                model_dir=work_dir, mode=ModeKeys.EVAL, no_collate=True),
    }
    trainer.preprocessor = preprocessor
    trainer.train()
