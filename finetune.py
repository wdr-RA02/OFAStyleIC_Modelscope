import argparse
from functools import partial

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.trainers.multi_modal import OFATrainer
from modelscope.utils.constant import ConfigKeys, ModeKeys
from modelscope.utils.hub import snapshot_download
from torchmetrics import PeakSignalNoiseRatio

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
    cfg.train.max_epochs=3
    return cfg

def preprocess_dataset(train_conf: dict,
                       remap: dict):
    '''
    生成hf格式的数据集

    args: train_conf: 训练配置文件
    remap: 重映射dict

    return: train_ds, eval_ds
    '''
    img_addr=train_conf["img_addr"]
    dataset_path=train_conf["dataset_path"]
    train_ds = generate_msdataset(dataset_path,
                                train_conf["train_json"],
                                remap)
    eval_ds = generate_msdataset(dataset_path, 
                                 train_conf["val_json"],
                                 remap)
    collate_fn=partial(collate_pcaption_dataset, dataset_dir=img_addr)
    # 处理数据集映射
    train_ds = train_ds.map(collate_fn)
    eval_ds = eval_ds.map(collate_fn)
    # print(train_ds[0])

    return train_ds, eval_ds

def generate_preprocessors(train_conf: dict,
                           work_dir: str,
                           tokenized: bool = False):
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

    if tokenized:
        # load style_dict
        # raise NotImplementedError("该部分尚未完工")
        style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
        style_dict=get_style_dict(style_list)

        print(style_dict)
        # add style token to tokenizers
        preprocessor[ConfigKeys.train].preprocess.add_style_token(style_dict)
        preprocessor[ConfigKeys.val].preprocess.add_style_token(style_dict)

    return preprocessor


def train(train_conf: dict, 
          train_ds,
          eval_ds,
          tokenize: bool,
          ckpt: str=None,
          work_dir: str="work_dir"):

    model_name=train_conf["model_name"]
    # model_dir = snapshot_download(model_name)
    # model_dir="workspace"
    # set dataset addr
    args = dict(
        model=model_name, 
        model_revision=train_conf["model_revision"],
        work_dir=work_dir, 
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg_modify_fn=cfg_modify_fn,
    )
    trainer = build_trainer(name=Trainers.ofa, default_args=args)

    # 为保安全加上这条assert
    assert type(trainer)==OFATrainer
    work_dir = args.get("work_dir", "workspace")

    trainer.preprocessor = generate_preprocessors(train_conf,
                                                  work_dir=work_dir,
                                                  tokenized=tokenize)
    if ckpt is None:
        print("No checkpoint, train from scratch.")
        trainer.train()
    else:
        print("checkpoint dir: "+ckpt) 
        trainer.train(checkpoint_path=ckpt)

def evaluate(train_conf: dict, 
             eval_ds,
             tokenize:bool,
             work_dir:str = "work_dir"):
    # model_dir = snapshot_download(model_name)
    model_dir=os.path.join(work_dir, "output")
    # set dataset addr
    args = dict(
        model=model_dir, 
        model_revision=train_conf["model_revision"],
        train_dataset=eval_ds,
        eval_dataset=eval_ds,
        cfg_modify_fn=cfg_modify_fn,
    )

    trainer = build_trainer(name=Trainers.ofa, default_args=args)
    trainer.preprocessor=generate_preprocessors(train_conf,
                                                work_dir=work_dir,
                                                tokenized=tokenize)
    print(trainer.evaluate())

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="OFA Style finetune tokenized")
    parser.add_argument("mode", help="select mode", choices=["train", "eval"])
    parser.add_argument("--trainer_conf", help="trainer config json", type=str, default="trainer_config.json")
    parser.add_argument("--checkpoint", help="checkpoint", type=str)
    args=parser.parse_args()

    # load args from config file
    train_conf=load_train_conf(args.trainer_conf)
    assert isinstance(train_conf, dict)

    work_dir=train_conf["work_dir"]
    tokenize=train_conf["tokenize_style"]
    ckpt=args.checkpoint
    print("work_dir is: "+work_dir)

    remap={
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    }

    train_ds, eval_ds=preprocess_dataset(train_conf, remap)
    if args.mode == "train":
        train(train_conf=train_conf, 
            train_ds=train_ds, 
            eval_ds=eval_ds, 
            tokenize=tokenize,
            work_dir=work_dir)
    elif args.mode == "eval":
        evaluate(train_conf=train_conf,
                 eval_ds=eval_ds,
                 tokenize=tokenize,
                 work_dir=work_dir)
