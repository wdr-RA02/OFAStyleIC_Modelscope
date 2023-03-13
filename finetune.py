from functools import partial

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.trainers.multi_modal import OFATrainer
from modelscope.utils.constant import ConfigKeys, ModeKeys
from modelscope.utils.hub import snapshot_download

from preprocessors.stylish_image_caption import OfaPreprocessorforStylishIC
from utils.build_dataset import generate_msdataset, collate_pcaption_dataset
from utils.train_conf import *


def cfg_modify_fn(cfg):
    cfg.train.hooks = [{
        'type': 'CheckpointHook',
        'by_epoch': False,
        'interval': 5000,
        'max_checkpoint_num': 3
    }, {
        'type': 'TextLoggerHook',
        'interval': 1
    }, {
        'type': 'IterTimerHook'
    }]
    cfg.train.max_epochs=2
    return cfg

def train():
    train_conf=load_train_conf("trainer_config.json")
    assert isinstance(train_conf, dict)

    remap={
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    }

    img_addr=train_conf["img_addr"]
    dataset_path=train_conf["dataset_path"]
    train_ds = generate_msdataset(dataset_path,
                                train_conf["train_json"],
                                remap)
    eval_ds = generate_msdataset(dataset_path, 
                                 train_conf["val_json"],
                                 remap)

    # 处理数据集映射
    collate_fn=partial(collate_pcaption_dataset, dataset_dir=img_addr)
    train_ds = train_ds.map(collate_fn)
    # print(train_ds[0])

    model_name=train_conf["model_name"]
    # model_dir = snapshot_download(model_name)
    model_dir="workspace"
    # set dataset addr
    args = dict(
        model=model_name, 
        model_revision=train_conf["model_revision"],
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
                model_dir=work_dir,
                mode=ModeKeys.TRAIN, 
                no_collate=True),
        ConfigKeys.val:
            OfaPreprocessorforStylishIC(
                model_dir=work_dir, 
                mode=ModeKeys.EVAL, 
                no_collate=True),
    }
    trainer.preprocessor = preprocessor
    trainer.train()



if __name__=="__main__":
    # load args from config file
    style_list=list_styles("personality.txt")
    style_dict=add_style_token(style_list)
    print(style_dict)

    work_dir="workspace"

    # train()


    
