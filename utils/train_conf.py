import os
import json

def load_train_conf(train_conf_filename: str):
    '''
    加载训练文件配置json

    args: train_conf_filename: 配置文件名称, 默认为trainer_config.json

    return: json化后的配置文件字典
    '''
    # 若加载失败则返回none
    train_conf = None
    with open(train_conf_filename) as f:
        train_conf=json.load(f)
    return train_conf

def list_styles(dataset_path: str,
                style_file_name: str):
    '''
    从personality_captions中加载风格列表
    
    args: style_file_name: personality.txt

    return: style_list
    '''

    style_list_dir=os.path.join(dataset_path, style_file_name)
    with open(style_list_dir) as f:
        style_list=f.read().strip("\n").split("\n")
    
    return style_list

def get_style_dict_from_ls(style_list: list):
    '''
    需要的时候调用此函数将personality_captions的风格逐一添加到self.tokenizer里面
    '''
    # 暂时没想到怎么加style_k, 先用code_k顶一下吧, 哎...
    style_dict = {style:f"<code_{i}>" for i, style in enumerate(style_list)}

    return style_dict

def generate_style_dict(train_conf: dict):
    '''
    根据train_conf直接返回需要的style_dict
    '''
    style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
    return get_style_dict_from_ls(style_list)


# config_modify_function
def cfg_modify_fn(args):
    # load arguments from args
    max_epoches:int=getattr(args, "max_epoches", 3)
    num_workers:int=getattr(args,"num_workers",0)
    batch_size:int=args.batch_size
    patch_img_size:int=args.patch_image_size
    max_img_size:int=args.max_image_size
    # hyper-parameters
    lr=getattr(args,"lr",None)
    lr_end=getattr(args,"lr_end",None)
    weight_decay=getattr(args, "weight_decay", None)
    warm_up=getattr(args,"warm_up", None)

    # ckpt hook may be changed based on whether scst is adpoted
    by_epoch: bool=args.cider if hasattr(args, "cider") else False
    ckpt_hook={
            'type': 'CheckpointHook',
            'by_epoch': by_epoch,
            'interval': [5000,1][by_epoch],
            'max_checkpoint_num': 3
        }
    
    train_conf=load_train_conf(args.conf)
    prompt=train_conf.get("prompt", None)
    # save all the logs and anything else to {work_dir}/miscs

    def mod_fn(cfg):
        # required by p_cap
        # add prompt config
        if prompt is not None and len!="":
            cfg.merge_from_dict({"model.prompt": prompt})
        cfg.model.patch_image_size=patch_img_size
        cfg.model.max_image_size=max_img_size
        # config adam begin lr        
        cfg.train.hooks = [
            ckpt_hook, 
        {
            'type': 'TextLoggerHook',
            'interval': 1
        }, {
            'type': 'IterTimerHook'
        }]
        cfg.train.max_epochs=max_epoches
        cfg.train.work_dir=train_conf["work_dir"]
        # hyper param
        if warm_up is not None:
            cfg.train.lr_scheduler.warmup_proportion=warm_up
        if lr_end is not None:
            cfg.train.lr_scheduler.lr_end=lr_end
        if lr is not None:
            cfg.train.optimizer.lr=lr
        if weight_decay is not None:
            cfg.train.optimizer.weight_decay=weight_decay
        # set up batch and workers
        cfg.train.dataloader.batch_size_per_gpu=batch_size
        cfg.train.dataloader.workers_per_gpu=num_workers
        cfg.evaluation.dataloader.batch_size_per_gpu=batch_size
        cfg.evaluation.dataloader.workers_per_gpu=num_workers
        # specify the eval metric
        cfg.evaluation.metrics=[{"type":"image-caption-metric"}]
        return cfg
    return mod_fn
