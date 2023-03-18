# OFAStyleIC_Modelscope
对Modelscope中OFA image caption的一系列组件进行适当修改，使得任务包含风格属性

## How to train?
Simple! 只需要在terminal中输入

```sh
CUDA_VISIBLE_DEVICES=x python3 finetune.py train (--conf path/to/conf.json)
```

然后坐等训完

~~基本上我用1080ti finetune一个OFA_tiny 2epoch都要12h....~~

反转了, 是我没调batch和num_worker捏

## Train script arguments
| args                             | help                       | default             |
|----------------------------------|-----------------------------|----------------------|
| -c/--conf path/to/train_conf     | 指定train configuration json | *required param*    |
| -p/--checkpoint path/to/ckpt     | 从指定的checkpoint开始训练   | None                |
| -e/--max_epoches N               | 最多训练多少epoch            | 3                   |
| -b/--batch_size N                | batch大小                    | 4                   |
| -w/--num workers N               | dataloader worker个数        | 0                   |

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
- damo/ofa_image-caption_coco_distilled_en, v1.0.1  [Modelscope](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary)  |  [conf json](conf_examples/distilled_tokenized.json)
- damo/ofa_pretrain_base_en, v1.0.2  [Modelscope](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary)  |  [conf json](conf/base_tokenized.json)

## How to inference?
Also simple! 
```sh
CUDA_VISIBLE_DEVICES=x python3 pre_inference.py --conf path/to/conf.json
```

推理结果保存了一份在{work_dir}/inference_{time}.json里

文件结构:
```py
{
    "model_name": "damo/ofa_pretrain_tiny_en",  # base模型名字
    "model_revision": "v1.0.1",
    "n_eval_sample": 10,                        # 样本个数
    "results": [                     # list, 包含每个样本的reference cap和生成cap
        {
            "style":"Style",
            "image":"path/to/img",
            "caption":"xxx",
            "reference":"yyy",
        },...
    ]
}
```

## Inference script arguments
| args                         | help                       | default             |
|-------------------------------|-----------------------------|----------------------|
| -c/--conf path/to/train_conf | 指定train configuration json | *required param* |
| -b/--batch_size N            | 测试的batch大小               | 10                  |

## How to save model?
shell在代码根目录下输入: 
```sh
python3 -m utils.backup_model --conf path_to_conf
```
然后就可以在work_dir文件夹下找到包含模型和配置的tar.gz文件


## Credits
- [Personality_Captions](https://openaccess.thecvf.com/content_CVPR_2019/html/Shuster_Engaging_Image_Captioning_via_Personality_CVPR_2019_paper.html)

- [Modelscope](https://modelscope.cn)

- [OFA](https://github.com/OFA-Sys/OFA)
