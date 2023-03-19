from .utils import generate_preprocessors, cfg_modify_fn
from .utils.constants import *
from .utils.build_dataset import generate_train_eval_ds
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers
from modelscope.trainers.multi_modal import OFATrainer

def start_evaluate(train_conf: dict,
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
        work_dir=train_conf["work_dir"],
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn)
    )
    trainer=build_trainer(name=Trainers.ofa, default_args=args)
    assert type(trainer)==OFATrainer
    print(trainer.evaluate())

def evaluate(args: argparse.Namespace):
    train_conf=load_train_conf(args.conf)
    assert isinstance(train_conf, dict)

    work_dir=train_conf["work_dir"]
    print("work_dir is: "+work_dir)

    remap={
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    }
    # load datasets
    _, eval_ds=generate_train_eval_ds(train_conf, eval=True, remap=remap)
    # modify_function
    mod_fn=cfg_modify_fn(args.max_epoches, args.batch_size, args.num_workers)
    start_evaluate(train_conf, eval_ds, mod_fn)