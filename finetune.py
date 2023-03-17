import argparse
from functools import partial
from typing import Callable

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.trainers.multi_modal import OFATrainer
from modelscope.utils.constant import ConfigKeys, ModeKeys
from modelscope.utils.hub import snapshot_download

from preprocessors.stylish_image_caption import OfaPreprocessorforStylishIC
from utils.build_dataset import generate_train_eval_ds
from utils.train_conf import *
from metric.stylish_ic_metric import ImageCaptionMetric

def cfg_modify_fn(max_epoches:int=3,
                  batch_size:int=4,
                  num_workers:int=0):
    def mod_fn(cfg):
        # required by p_cap
        cfg.model.patch_image_size=224
        cfg.model.max_image_size=256        
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
        cfg.train.max_epochs=max_epoches
        # set up batch and workers
        cfg.train.dataloader.batch_size_per_gpu=batch_size
        cfg.train.dataloader.workers_per_gpu=num_workers
        cfg.evaluation.dataloader.batch_size_per_gpu=batch_size
        cfg.evaluation.dataloader.workers_per_gpu=num_workers
        # specify the eval metric
        cfg.evaluation.metrics=[{"type":"image-caption-metric"}]
        return cfg
    return mod_fn


def generate_preprocessors(train_conf: dict,
                           mod_fn: Callable = None,
                           tokenize: bool = False):
    '''
    生成preprocessor

    args:
    train_conf: train config字典
    tokenize: 是否将风格tokenize为<code_k>
    '''
    model_dir=os.path.join(train_conf["work_dir"], "output")
    model_dir=model_dir if os.path.exists(model_dir) \
                        else snapshot_download(train_conf["model_name"],
                                revision=train_conf["model_revision"])
    # 此时config还没有改过来
    preprocessor = {
        ConfigKeys.train:
            OfaPreprocessorforStylishIC(
                model_dir=model_dir,
                cfg_modify_fn=mod_fn,
                mode=ModeKeys.TRAIN, 
                no_collate=True),
        ConfigKeys.val:
            OfaPreprocessorforStylishIC(
                model_dir=model_dir, 
                cfg_modify_fn=mod_fn,
                mode=ModeKeys.EVAL, 
                no_collate=True)
    }

    if tokenize:
        # load style_dict
        style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
        style_dict=get_style_dict(style_list)

        # print(style_dict)
        # add style token to tokenizers
        preprocessor[ConfigKeys.train].preprocess.add_style_token(style_dict)
        preprocessor[ConfigKeys.val].preprocess.add_style_token(style_dict)

    return preprocessor


def generate_trainer(train_conf: dict, 
                     train_ds,
                     eval_ds,
                     tokenize: bool,
                     mod_fn: Callable):
    '''
    生成含有修改的trainer供训练或eval使用
    
    arg: 
    train_conf: train config字典
    train_ds, eval_ds: 数据集
    tokenize: 是否将风格tokenize为<code_k>
    mod_fn: cfg_modify_fn

    return: trainer
    '''
    model_name=train_conf["model_name"]
    work_dir=train_conf["work_dir"]
    # model_dir = snapshot_download(model_name)
    # set dataset addr
    args = dict(
        model=model_name, 
        model_revision=train_conf["model_revision"],
        work_dir=work_dir, 
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn,
                                            tokenize=tokenize)
    )
    trainer = build_trainer(name=Trainers.ofa, default_args=args)

    # 为保安全加上这条assert
    assert type(trainer)==OFATrainer
    if tokenize:
        style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
        style_dict=get_style_dict(style_list)
        trainer.model.tokenizer.add_tokens(list(style_dict.values()))
    
    return trainer

def train(trainer, 
          ckpt: str=None):
    if ckpt is None:
        print("No checkpoint, train from scratch.")
        trainer.train()
    else:
        print("checkpoint dir: "+ckpt) 
        trainer.train(checkpoint_path=ckpt)

def evaluate(train_conf: dict,
             eval_ds, 
             mod_fn: Callable):
    model_dir = os.path.join(train_conf["work_dir"],"output")
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Model dir {} not exist".format(model_dir))
    # set dataset addr
    args = dict(
        model=model_dir, 
        model_revision=train_conf["model_revision"],
        train_dataset=eval_ds,
        eval_dataset=eval_ds,
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn,
                                            tokenize=tokenize)
    )
    trainer=build_trainer(name=Trainers.ofa, default_args=args)
    assert type(trainer)==OFATrainer
    print(trainer.evaluate())

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="OFA Style finetune tokenized")
    parser.add_argument("mode", help="select mode", choices=["train", "eval"])
    parser.add_argument("-c", "--conf", help="trainer config json", type=str, required=True)
    parser.add_argument("-p", "--checkpoint", help="checkpoint", type=str)
    parser.add_argument("-e", "--max_epoches", help="max epoch", type=int, default=3)
    parser.add_argument("-b", "--batch_size", help="#samples per batch", type=int, default=4)
    parser.add_argument("-w","--num_workers", help="num of dataloader", type=int, default=0)
    args=parser.parse_args()

    # load args from config file
    train_conf=load_train_conf(args.conf)
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
    # load datasets
    train_ds, eval_ds=generate_train_eval_ds(train_conf, remap)
    # modify_function
    mod_fn=cfg_modify_fn(args.max_epoches, args.batch_size, args.num_workers)
    
    if args.mode == "train":
        trainer=generate_trainer(train_conf, 
                             train_ds, 
                             eval_ds, 
                             train_conf["tokenize_style"],
                             mod_fn)
        train(trainer, ckpt=ckpt)
    elif args.mode == "eval":
        evaluate(train_conf, eval_ds, mod_fn)