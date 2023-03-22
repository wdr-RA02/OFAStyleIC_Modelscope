from time import sleep
from .scst_criterion import SCSTCriterion
from .utils import generate_preprocessors
from .utils.constants import *
from .utils.build_dataset import generate_train_eval_ds
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers
from modelscope.trainers.multi_modal import OFATrainer
from modelscope.models.multi_modal import OfaForAllTasks

def generate_trainer(train_conf: dict, 
                     train_ds,
                     mod_fn: Callable):
    '''
    生成含有修改的trainer供训练或eval使用
    
    arg: 
    train_conf: train config字典
    train_ds, eval_ds: 数据集
    mod_fn: cfg_modify_fn

    return: trainer
    '''
    model_name=train_conf["model_name"]
    work_dir=train_conf["work_dir"]
    tokenize=train_conf["tokenize_style"]
    # model_dir = snapshot_download(model_name)
    # set dataset addr
    args = dict(
        model=model_name, 
        model_revision=train_conf["model_revision"],
        work_dir=work_dir, 
        train_dataset=train_ds,
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn)
    )
    trainer = build_trainer(name=Trainers.ofa, default_args=args)
    assert isinstance(trainer, OFATrainer)

    cider_debug=False
    if cider_debug:
        print("WARNING: CIDEr finetune is under test!!!")
        sleep(5)
        criterion_args=trainer.criterion.args
        trainer.criterion=SCSTCriterion(criterion_args)
    
    # froze the resnet
    assert hasattr(trainer, "model") and isinstance(trainer.model, OfaForAllTasks)
    for name, param in trainer.model.named_parameters():
        if "embed_images" in name:
            param.requires_grad=False

    if tokenize:
        style_dict=generate_style_dict(train_conf)
        trainer.model.tokenizer.add_tokens(list(style_dict.values()))
    
    return trainer

def start_train(trainer, 
          ckpt: str=None):
    if ckpt is None:
        print("No checkpoint, train from scratch.")
        trainer.train()
    else:
        print("checkpoint dir: "+ckpt) 
        trainer.train(checkpoint_path=ckpt)

def train(args: argparse.Namespace, mod_fn: Callable):
    train_conf=load_train_conf(args.conf)
    assert isinstance(train_conf, dict)

    work_dir=train_conf["work_dir"]
    ckpt=args.checkpoint
    print("work_dir is: "+work_dir)

    remap={
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    }
    # load datasets
    train_ds, _=generate_train_eval_ds(train_conf, train=True, remap=remap)
    trainer=generate_trainer(train_conf, 
                            train_ds, 
                            mod_fn)
    start_train(trainer, ckpt=ckpt)