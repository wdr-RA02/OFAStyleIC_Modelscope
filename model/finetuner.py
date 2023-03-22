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
                     mod_fn: Callable,
                     from_pretrained: str=None,
                     use_cider: bool=False):
    '''
    生成含有修改的trainer供训练或eval使用
    
    arg: 
    train_conf: train config字典
    train_ds, eval_ds: 数据集
    mod_fn: cfg_modify_fn
    use_cider: 是否使用scst作为训练任务

    return: trainer
    '''
    if from_pretrained is not None and isinstance(from_pretrained, str):
        model_dir=from_pretrained
        
        print("checkpoint dir: "+model_dir) 
        assert os.path.exists(os.path.join(model_dir, "pytorch_model.bin")), \
            "file pytorch_model.bin not found in {}".format(model_dir)
    else:
        print("No checkpoint, train from scratch.")
        model_dir=train_conf["model_name"]
    work_dir=train_conf["work_dir"]
    tokenize=train_conf["tokenize_style"]
    # model_dir = snapshot_download(model_name)
    # set dataset addr
    args = dict(
        model=model_dir, 
        model_revision=train_conf["model_revision"],
        work_dir=work_dir, 
        train_dataset=train_ds,
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn)
    )
    trainer = build_trainer(name=Trainers.ofa, default_args=args)
    assert isinstance(trainer, OFATrainer)

    if use_cider:
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

def train(args: argparse.Namespace, mod_fn: Callable, use_cider_scst: bool=False):
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
                            mod_fn,
                            use_cider=use_cider_scst,
                            from_pretrained=ckpt)
    trainer.train()