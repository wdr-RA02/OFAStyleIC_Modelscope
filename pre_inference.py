import argparse
from typing import Dict
from modelscope.pipelines import pipeline
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.models.multi_modal import OfaForAllTasks
# from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

from preprocessors.stylish_image_caption import OfaPreprocessorforStylishIC
from utils.build_dataset import generate_msdataset, collate_pcaption_dataset
from utils.train_conf import *

def get_batch_addr(base_addr:str, hashes: dict):
    batch_data=list()
    for hash in hashes:
        one_data = {
            "image": base_addr+hash+".jpg",
            "style": hashes[hash].lower()
        }
        batch_data.append(one_data)
    return batch_data

def inference_orig(train_conf_file: str,
                   data):
    train_conf=load_train_conf(train_conf_file)
    assert isinstance(train_conf, dict)

    # model_name="damo/ofa_image-caption_coco_distilled_en"
    # model_dir = snapshot_download(model_name)
    # img = "https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg"


    # model_dir = snapshot_download(model_name)
    model_dir=os.path.join(train_conf["work_dir"], "output")
    tokenize=train_conf["tokenize_style"]

    # define preprocessor and model
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir)
    model=OfaForAllTasks.from_pretrained(model_dir)
    if tokenize:
        style_list=list_styles(train_conf["dataset_path"], "personalities.txt")
        style_dict=get_style_dict(style_list)

        preprocessor.preprocess.add_style_token(style_dict)
        print(preprocessor.tokenizer.tokenize(["<code_1>"]))

    stylish_ic=pipeline(Tasks.image_captioning, 
                        model=model, 
                        preprocessor=preprocessor)
    
    # result=stylish_ic(img)
    result=stylish_ic(data)
    for i in result:
        print(i)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="OFA Style inference")
    parser.add_argument("--trainer_conf", help="trainer config json", type=str, default="trainer_config.json")
    args=parser.parse_args()

    train_conf_file=args.trainer_conf

    batches={
        "2923e28b6f588aff2d469ab2cccfac57":"Obsessive",
        "73a33823bb3e8ef618bf52f4b3147d":"Mellow (Soothing, Sweet)",
        "e7a8a76ea32c1117dde5b93f2e18e":"Sensitive",
        "f3d0f4eb52e6ee38c8b9cef1b6272":"Unimaginative",
        "29697d81476d0307376e7466f6ad48":"Casual"
    }

    addr=load_train_conf(train_conf_file)["img_addr"]
    data=get_batch_addr(addr,batches)

    inference_orig("conf/distilled_orig.json", data)

