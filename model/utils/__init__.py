from .constants import *
from preprocessor.stylish_image_caption import OfaPreprocessorforStylishIC

def cfg_modify_fn(max_epoches:int=3,
                  batch_size:int=4,
                  num_workers:int=0):
    def mod_fn(cfg):
        # required by p_cap
        cfg.model.patch_image_size=224
        cfg.model.max_image_size=256
        # config adam begin lr        
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
                           mod_fn: Callable = None):
    '''
    生成preprocessor

    args:
    train_conf: train config字典
    tokenize: 是否将风格tokenize为<code_k>
    '''
    model_dir=os.path.join(train_conf["work_dir"], "output")
    model_dir=model_dir if os.path.exists(os.path.join(model_dir),"pytorch_model.bin") \
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

    if train_conf["tokenize_style"]:
        # load style_dict
        style_dict=generate_style_dict(train_conf)
        # print(style_dict)
        # add style token to tokenizers
        preprocessor[ConfigKeys.train].preprocess.add_style_token(style_dict)
        preprocessor[ConfigKeys.val].preprocess.add_style_token(style_dict)

    return preprocessor