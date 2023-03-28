# OFAStyleIC_Modelscope
对Modelscope中OFA image caption的一系列组件进行适当修改，使得任务包含风格属性

## Updates
- 2023-03-19: finetune.py和pre_inference.py已被重构成models.{finetuner, evaluator, inference_pipeline}三个模块, 并由model_operator.py统一调用
- 2023-03-22: 实验性地加入scst功能(详见func/add_scst分支)
- 2023-03-26: 添加环境配置章节
- 2023-03-27: inference支持指定推理样本, 添加json文件样例

## Env setup
这里推荐使用conda
```sh
conda create -n myenv python=3.8.5
conda activate myenv

conda install jq
conda install pytorch==1.12.1 torchaudio==0.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
pip3 install -r requirements.txt
```

## How to use
### Train/Eval:
若想并卡, 则使用:
```shell
CUDA_VISIBLE_DEVICES=x,y,... torchrun --rdzv_backend c10d \
        --rdzv_endpoint localhost:30000 \
        --nnodes 1 --nproc_per_node N \ 
        model_operator.py train/eval --conf path/to/conf.json ...
```
> **注意, 由于修改了代码, 所以就算是单卡也必须这么运行**


然后坐等训完, 训练好的模型会存放在{work_dir}/output里面
测试指标: BLEU-1, BLEU-4, ROUGE-L, CIDEr, SPICE

### trainer_conf.json结构
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
    "prompt":带{}的prompt, 不指定的话填null/""
}
```
我提供了几个配置例子:
- [base_tokenized_pt](conf_examples/base_tokenized_pt.json)
- [tiny_tokenized_pt](conf_examples/tiny_tokenized_pt.json)
- [distilled_tokenized_coco](conf_examples/distilled_tokenized_coco.json)

### Inference:
Also simple! 
```shell
CUDA_VISIBLE_DEVICES=x python3 model_operator.py inference \
                                 --conf path/to/conf.json [-r|-j PATH/TO/JSON]
```

当使用-r时, 会从eval dataset里随机抽取b份样本进行推理

当然, 也可以自己组织一份json文件, 指定一些image_hash进行测试. 以下给出json文件[范例](./conf_examples/inference_base_a.json):
```python
[
    {
        "style": "Destructive",
        "image_hash": "17bb5c2fddbd6ffcd4d35d43755cadd",
        # 或: 
        #"image": "path/to/yfcc_img/17bb5c2fddbd6ffcd4d35d43755cadd.jpg",
        # 二者皆可, 如果给了image_hash那么inference_pipeline会自己添加上yfcc目录
        "reference": "some_reference"
        # style和image/image_hash是必须有的, reference可有可无, 主要取决于需不需要参考
        # 如果给定了这一行那么输出的json文件也会有
    },
    ...
]

```

推理结果保存了一份在{work_dir}/inference_{time}.json里面, 文件结构:
```python
{
    "model_name": "damo/ofa_pretrain_tiny_en",  # base模型名字
    #这一栏可以通过-d修改, 详见参数节
    "model_revision": "v1.0.1",
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

## Model_operator.py arguments
| args                           | help                                        | default    | available in          |
|--------------------------------|---------------------------------------------|:----------:|:---------------------:|
| -c/--conf path/to/train_conf   | 指定train configuration json                  | *required* | ALL                   |
| -b/--batch_size N              | batch大小                                     | 4          | ALL                   |
| -p/--patch_image_size P        | resnet patch大小                              | 224        | ALL                   |
| -m/--max_image_size M          | resnet max image大小                          | 256        | ALL                   |
| --cider                        | 是否进行基于CIDEr的scst优化 *1 *2              | False      | ``train``             |
| -e/--max_epoches N             | 最多训练多少epoch                                 | 3          | ``train``             |
| -t/--checkpoint path/to/ckpt   | 指定ckpt目录                                    | None       | ``train``             |
| --lr LR                        | 调整学习率                                       | 5e-5       | ``train``             |
| --lr_end LR_END                | 调整学习率终点                                     | 1e-7       | ``train``             |
| --warm_up W_UP                 | 调整warmup rate                               | 0.01       | ``train``             |
| --weight_decay W_DECAY         | 调整weight decay rate                         | 0.001      | ``train``             |
| --beam_size                    | 调整beam size                                 | 5          | ``train``             |
| --max_len                      | 调整max length                                | 16         | ``train``             |
| --freeze_resnet                | 训练时是否冻结ResNet                               | False      | ``train``             |
| -l/--log_csv_file CSV_DIR      | 评估时将指标存在CSV_DIR中                            | None       | ``eval``              |
| -w/--num workers N             | dataloader worker个数                         | 0          | ``train`` && ``eval`` |
| -r/--random                    | 从eval中直接抽取样本进行推理测试                          | False      | ``inference``         |
| -j/--inference_json JSON *3    | 从预先准备好的json文件中提取样本进行推理测试                    | None       | ``inference``         |
| -d/--description DESC *3       | 修改输出的推理文件中``model_name``这一字段的内容, 而不使用模型名称 | None       | ``inference``         |
| -o/--out_dir  OUT_DIR          | 指定推理json文件的输出目录<br>默认输出到``{work_dir}``下         | None       | ``inference``         |

注: 
> 1: SCST功能仍是早期版本, 相当不稳定! 
> 
> 2: 使用SCST时必须指定checkpoint 
> 
> 3, 4: 推理时, -r和-j必须选择一个

我目前准备使用的模型:
- damo/ofa_image-caption_coco_distilled_en, v1.0.1  [Modelscope](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary)  |  [conf json](conf_examples/distilled_tokenized.json)
- damo/ofa_pretrain_base_en, v1.0.2  [Modelscope](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary)  |  [conf json](conf/base_tokenized.json)


## How to save model?
shell在代码根目录下输入: 
```sh
python3 -m utils.backup_model --conf path_to_conf
```
然后就可以在work_dir文件夹下找到包含模型和配置的tar.gz文件

## utils.backup_model arguments
| args                         | help                         | default          |
|------------------------------|------------------------------|:------------------:|
| -c/--conf path/to/train_conf | 指定train configuration json   | *required* |
| -o/--out_dir OUT_DIR         | 指定输出目录                       | ./work_dir       |
| -e/--example_json            | 若使用该参数, 则会保存抹去dataset位置的json | False            |
| -n/--no_json                 | 若使用该参数, 则打包时会忽略配置json文件      | False            |

> 注意: -e/-n只能二择其一


## Credits
- [Personality_Captions](https://openaccess.thecvf.com/content_CVPR_2019/html/Shuster_Engaging_Image_Captioning_via_Personality_CVPR_2019_paper.html)

- [Modelscope](https://modelscope.cn)

- [OFA](https://github.com/OFA-Sys/OFA)

- pycocoevalcap: [1](https://github.com/salaniz/pycocoevalcap)  &  [2](https://github.com/tylin/coco-caption)
