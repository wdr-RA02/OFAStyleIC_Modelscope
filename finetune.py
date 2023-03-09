from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from preprocessors.stylish_image_caption import OfaPreprocessorforStylishIC
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ModeKeys

import os

def generate_msdataset(ds_path: str, json_name: str):
    """load pcaptions datasets from the given json"""
    ds=MsDataset.load('json',
                    data_files=os.path.join(ds_path, json_name))
    ds=ds.remap_columns({
        "personality":"style",
        "comment":"text",
        "image_hash":"image_hash"
    })
    return ds


if __name__=="__main__":
    img_addr="xxx"
    dataset_path="D:/study_local/final_design_code/personality_captions/"
    dataset_train = generate_msdataset(dataset_path, "train.json")
    print(next(iter(dataset_train)))

    model_name="damo/ofa_image-caption_coco_distilled_en"
    model_dir = snapshot_download(model_name)
    preprocessor=OfaPreprocessorforStylishIC(model_dir=model_dir, 
                                             mode=ModeKeys.TRAIN,
                                             dataset_dir=img_addr,
                                             dataset_file_attr=".jpg")
    
    print(preprocessor(next(iter(dataset_train))))

