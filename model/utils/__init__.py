from .constants import *
from .config_modify_fn import cfg_modify_fn
from preprocessor.stylish_image_caption import OfaPreprocessorforStylishIC

def generate_preprocessors(train_conf: dict,
                           from_pretrained: str=None,
                           mod_fn: Callable = None):
    '''
    生成preprocessor

    args:
    train_conf: train config字典
    from_pretrained: str 提供checkpoint的位置, 要求目录下存在pytorch_model.bin文件
    tokenize: 是否将风格tokenize为<code_k>
    '''
    if from_pretrained is not None and isinstance(from_pretrained, str):
        model_dir=from_pretrained
        print("checkpoint dir: "+model_dir) 
        assert os.path.exists(os.path.join(model_dir, "pytorch_model.bin")), \
            "file pytorch_model.bin not found in {}".format(model_dir)
    else:
        print("No checkpoint")
    model_dir=os.path.join(train_conf["work_dir"], "output")
    model_dir=model_dir if os.path.exists(os.path.join(model_dir,"pytorch_model.bin")) \
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