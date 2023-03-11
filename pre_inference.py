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

if __name__ == "__main__":
    train_conf=load_train_conf("trainer_config.json")
    assert isinstance(train_conf, dict)

    # model_name="damo/ofa_image-caption_coco_distilled_en"
    # model_dir = snapshot_download(model_name)
    # img = "https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg"

    addr=train_conf["img_addr"]
    model_name=train_conf["model_name"]
    # model_dir = snapshot_download(model_name)
    model_dir="workspace"

    # define preprocessor and model
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir)
    model=OfaForAllTasks.from_pretrained(model_dir)

    batches={
        "2923e28b6f588aff2d469ab2cccfac57":"Obsessive",
        "73a33823bb3e8ef618bf52f4b3147d":"Mellow (Soothing, Sweet)",
        "e7a8a76ea32c1117dde5b93f2e18e":"Sensitive",
        "f3d0f4eb52e6ee38c8b9cef1b6272":"Unimaginative",
        "29697d81476d0307376e7466f6ad48":"Casual"
    }
    stylish_ic=pipeline(Tasks.image_captioning, 
                        model=model, 
                        preprocessor=preprocessor)
    
    data=get_batch_addr(addr,batches)
    # result=stylish_ic(img)
    result=stylish_ic(data)
    print(result)