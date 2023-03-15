# OFAStyleIC_Modelscope
对Modelscope中OFA image caption的一系列组件进行适当修改，使得任务包含风格属性

## trainer_conf.json结构
```py
{
    "model_name":"要finetune的模型名称",
    "model_revision": "模型版本",
    "tokenize_style": true/false,          # 是否对风格进行tokenize处理
    "img_addr":"path_to_PIC/yfcc_images/", # 务必记得加最后那个/
    "dataset_path":"path_to_PIC/personality_captions/",
    "file_attr":".jpg",
    "train_json":"train.json",
    "test_json":"test.json",
    "val_json":"val.json",
    "work_dir":"path_to_work_dir/",
    "prompt":"" #目前还没实现这个功能
}
```
训练好的模型会存放在{work_dir}/output里面

我目前准备使用的模型:
- [damo/ofa_image-caption_coco_distilled_en, v1.0.1](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary)
- [damo/ofa_pretrain_base_en, v1.0.2](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary)

## How to train?
python3 finetune.py train (--trainer_conf path/to/conf.json)

然后坐等训完捏

基本上我用1080ti finetune一个OFA_tiny 2epoch都要12h....


## Credits
- [Personality_dataset](https://openaccess.thecvf.com/content_CVPR_2019/html/Shuster_Engaging_Image_Captioning_via_Personality_CVPR_2019_paper.html)

- [Modelscope](https://modelscope.cn)

- [OFA](https://github.com/OFA-Sys/OFA)
