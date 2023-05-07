import random
from datetime import datetime as dt
from modelscope.pipelines import pipeline


from metric import stylish_ic_metric
from .utils.constants import *
from .utils.build_dataset import generate_msdataset, collate_pcaption_dataset
from preprocessor import OfaPreprocessorforStylishIC


def get_eval_batch(train_conf: dict,
                   data: list):
    # use collate_pcaption fn to add address
    if "image_hash" in data[0]:
        for item in data:
            item["image"]=item.pop("image_hash")
    out_data=list(map(lambda x:collate_pcaption_dataset(x, 
                                                   train_conf["img_addr"], 
                                                   train_conf["file_attr"]), data))
    return out_data


def generate_infr_pipeline(train_conf: Dict[str, str], 
                           mod_fn: Callable):
    model_dir=os.path.join(train_conf["work_dir"], "output")
    tokenize=train_conf["tokenize_style"]
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir, cfg_modify_fn=mod_fn)
    if tokenize:
        style_dict=generate_style_dict(train_conf)
        assert isinstance(style_dict, dict)
        print("Tokenize style=True, add style dict. ")
        preprocessor.preprocess.add_style_token(style_dict)

    stylish_ic=pipeline(Tasks.image_captioning, 
                        model=model_dir, 
                        preprocessor=preprocessor)

    return stylish_ic


def start_inference_from_json(train_conf: Dict[str,str],
                              json_file: str, 
                              mod_fn: Callable):
    # define preprocessor and model
    stylish_ic=generate_infr_pipeline(train_conf, mod_fn)

    with open(json_file, "r") as f:
        # needs to follow format of {[{"image","style",("reference")}]}
        data=json.load(f)
        if "image_hash" in data[0]:
            data=get_eval_batch(train_conf,data)
        elif "image" not in data[0]:
            raise KeyError("json file missing key 'image'")

    result=[]
    result_cap=list(map(lambda x:x.get(OutputKeys.CAPTION), stylish_ic(data)))
    for i in range(len(result_cap)):
        # add original text and style to captions
        result.append({
            **data[i],
            "caption": result_cap[i][0]
        })
    return result


def start_inference_from_eval(train_conf: dict,
                   data: List[Dict[str,str]],
                   mod_fn: Callable):
    '''
    data: ["style", "image", "text"]
    '''
    # save ground truth and pop it for the pipeline
    orig_text=list(map(lambda x:{"reference":x.pop("text")},data))

    # define preprocessor and model
    stylish_ic=generate_infr_pipeline(train_conf, mod_fn)


def start_inference_from_eval(train_conf: dict,
                   data: List[Dict[str,str]],
                   mod_fn: Callable):
    '''
    data: ["style", "image", "text"]
    '''
    # save ground truth and pop it for the pipeline
    orig_text=list(map(lambda x:{"reference":x.pop("text")},data))

    # define preprocessor and model
    stylish_ic=generate_infr_pipeline(train_conf, mod_fn)

    result=[]
    result_cap=list(map(lambda x:x.get(OutputKeys.CAPTION), stylish_ic(data)))
    for i in range(len(result_cap)):
        # add original text and style to captions
        result.append({
            **data[i],
            **orig_text[i],
            "caption": result_cap[i][0],
        })
    return result

def save_results_to_json(train_conf: Dict[str, str],
                         result: Dict[str, Any],
                         description: str=None,
                         output_dir: str=None):
    '''
    将result保存到{work_dir}/inference_{dt}.json中,
    包含: model_name, result{image_path, caption, reference}
    
    args:
    train_conf: 训练配置
    result: 推理结果
    description: 保存到model_name字段的内容
    output_dir: 推理结果json保存位置
    '''
    name=description if description is not None else train_conf["model_name"]
    result={
        "model_name": name,
        "model_revision": train_conf["model_revision"],
        "results":result
    }
    # save result to work_dir
    result_filename="inference_{}.json".format(dt.strftime(dt.now(), "%y%m%d-%H%M%S"))
    if output_dir is None:
        output_dir=train_conf["work_dir"]
    else:
        os.makedirs(output_dir, exist_ok=True)
    result_filename=os.path.join(output_dir, result_filename)
    with open(result_filename, "w") as f:
        json.dump(result, f, indent=4)
    print("Inference also saved to {}".format(result_filename))


def inference(args: argparse.Namespace, mod_fn):
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
    if args.random:
    # randomly select 10 samples from val set
        batches=random.choices(eval_ds, k=args.batch_size)
    # style_dict
        data=get_eval_batch(train_conf, batches)
        result=start_inference_from_eval(train_conf, data, mod_fn)
    else:
        if hasattr(args, "batch_size"):
            print("WARNING: -b will lose function when -j in place")
        result=start_inference_from_json(train_conf, args.inference_json, mod_fn)

    print(*result)
    # save model info to json file
    save_results_to_json(train_conf, result,
                         description=getattr(args, "description", None),
                         output_dir=getattr(args, "out_dir", None))
