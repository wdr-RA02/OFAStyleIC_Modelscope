## 模型信息:
- 基于OFA_...预训练
- 风格tokenize为<code_i>
- 其他训练任务: *若有请列出*

## 指标:
| metric  | score |
|---------|-------|
| BLEU-1  | x.y   |
| BLEU-4  | x.y   |
| Rouge-L | x.y   |
| CIDEr  | x.y   |
| SPICE  | x.y   |

## 如何使用:
- clone代码, 在代码根目录解压模型
- 进入conf/.json文件指定PCap目录
- 开始使用

----------------------------------

## Model properties:
- pretrained on OFA_... model
- tokenized style with <code_i>
- Other finetune missions: *specify if any*

## Metrics: 
refer to table above

## How_to:
- download code, and untar the archive there
- specify PCap dir in conf/{}.json
- Start to use

> model options: (be sure to delete this in official release note)
>- ofa_pretrain_base_en
>- ofa_pretrain_tiny_en
>- ofa_image-caption_coco_distilled_en