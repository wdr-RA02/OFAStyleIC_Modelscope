import random
from datetime import datetime as dt
from modelscope.pipelines import pipeline
from .utils.constants import *
from .utils.build_dataset import generate_msdataset, collate_pcaption_dataset
from preprocessor.stylish_image_caption import OfaPreprocessorforStylishIC


def get_eval_batch(train_conf: dict,
                   data: list):
    # use collate_pcaption fn to add address
    out_data=list(map(lambda x:collate_pcaption_dataset(x, 
                                                   train_conf["img_addr"], 
                                                   train_conf["file_attr"]), data))
    return out_data


def start_inference(train_conf: dict,
                   data: List[Dict[str,str]]):
    '''
    data: ["style", "image", "text"]
    '''
    # model_dir = snapshot_download(model_name)
    model_dir=os.path.join(train_conf["work_dir"], "output")
    tokenize=train_conf["tokenize_style"]

    # save ground truth and pop it for the pipeline
    orig_text=list(map(lambda x:{"reference":x.pop("text")},data))

    # define preprocessor and model
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir)
    if tokenize:
        style_dict=generate_style_dict(train_conf)
        assert isinstance(style_dict, dict)
        print("Tokenize style=True, add style dict. ")
        preprocessor.preprocess.add_style_token(style_dict)

    stylish_ic=pipeline(Tasks.image_captioning, 
                        model=model_dir, 
                        preprocessor=preprocessor)
    
    result=[]
    result_cap=list(map(lambda x:x.get(OutputKeys.CAPTION), stylish_ic(data)))
    for i in range(len(result_cap)):
        # add original text and style to captions
        result.append({
            **data[i],
            "caption": result_cap[i][0],
            **orig_text[i],

        })
    return result


def inference(args: argparse.Namespace):
    train_conf=load_train_conf(args.conf)

    # load eval dataset\
    remap={
        "personality":"style",
        "comment":"text",
        "image_hash":"image"
    }
    eval_ds=generate_msdataset(ds_path=train_conf["dataset_path"], 
                               json_name=train_conf["val_json"],
                               remap_dict=remap)
    # randomly select 10 samples from val set
    batches=random.choices(eval_ds, k=args.batch_size)
    # style_dict
    data=get_eval_batch(train_conf, batches)
    result=start_inference(train_conf, data)
    print(*result)
    # add model info
    result={
        "model_name": train_conf["model_name"],
        "model_revision": train_conf["model_revision"],
        "n_eval_sample": args.batch_size,
        "results":result
    }
    # save result to work_dir
    result_filename="inference_{}.json".format(dt.strftime(dt.now(), "%y%m%d-%H%M%S"))
    result_filename=os.path.join(train_conf["work_dir"], result_filename)
    with open(result_filename, "w") as f:
        json.dump(result, f, indent=4)
    print("Inference also saved to {}".format(result_filename))