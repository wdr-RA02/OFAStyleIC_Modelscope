# config_modify_function
def cfg_modify_fn(args,
                  prompt: str=None):
    # load arguments from args
    max_epoches:int=args.max_epoches if hasattr(args, "max_epoches") else 3
    num_workers:int=args.num_workers if hasattr(args, "num_workers") else 0
    batch_size:int=args.batch_size
    patch_img_size:int=args.patch_img_size
    max_img_size:int=args.max_img_size

    def mod_fn(cfg):
        # required by p_cap
        # add prompt config
        if prompt is not None and len!="":
            cfg.merge_from_dict({"model.prompt": prompt})
        cfg.model.patch_image_size=patch_img_size
        cfg.model.max_image_size=max_img_size
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
