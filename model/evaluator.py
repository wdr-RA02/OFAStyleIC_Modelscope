from .utils import generate_preprocessors
from .utils.constants import *
from .utils.build_dataset import generate_ready_ds
from modelscope.utils.config import Config
from modelscope.trainers import build_trainer
from modelscope.metainfo import Trainers
from modelscope.trainers.multi_modal import OFATrainer
import csv
from datetime import datetime as dt

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
        cfg_modify_fn=mod_fn,
        preprocessor=generate_preprocessors(train_conf,
                                            mod_fn=mod_fn)
    )
    trainer=build_trainer(name=Trainers.ofa, default_args=args)
    assert type(trainer)==OFATrainer
    results=trainer.evaluate()
    results={k:"{.2f}".format(round(v*100)) for k,v in results.items()}
    print(results)

    return results

def result_to_csv(train_conf: dict, 
                  results: dict,
                  save_csv_filename: str):
    '''
    将result写入csv文件方便进行指标对比
    写入内容: eval_time, params, results

    args:
    train_conf: 训练配置
    results: trainer.evaluate()的输出
    save_csv_filename: 保存csv的名称, 由arg指定

    '''
    if save_csv_filename is None:
        return
    
    json_dir=os.path.join(train_conf["work_dir"],"output", "configuration.json")
    cfg=Config.from_file(json_dir)
    # get necessary args
    params=dict(
        work_dir=train_conf["work_dir"],
        epoches=str(cfg.train.max_epochs),
        warm_up=str(cfg.train.lr_scheduler.warmup_proportion),
        lr=str(cfg.train.optimizer.lr),
        lr_end=str(cfg.train.lr_scheduler.lr_end),
        weight_decay=str(cfg.train.optimizer.weight_decay),
    )
    # create title bar if not exist
    titles=["eval_time", *params.keys(), *results.keys()]
    csv_exists=os.path.exists(save_csv_filename)
    with open(save_csv_filename, 'a+', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=titles)
        if not csv_exists:
            writer.writeheader()
        # write train_time, param and results
        writer.writerow({
            "eval_time": dt.strftime(dt.now(),"%Y%m%d-%H%M%S"),
            **params,
            **results
        })
    
    print("This result saved to {}".format(save_csv_filename))
    
    
def evaluate(args: argparse.Namespace, mod_fn: Callable):
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
    eval_ds=generate_ready_ds(train_conf, ds_type="test", remap=remap)
    # modify_function
    results=start_evaluate(train_conf, eval_ds, mod_fn)

    csv_filename=getattr(args,"log_csv_file",None)
    result_to_csv(train_conf, results, csv_filename)