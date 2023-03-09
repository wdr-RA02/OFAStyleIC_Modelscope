from asyncio.windows_utils import pipe
import modelscope
from modelscope.pipelines import pipeline
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from .preprocessors.OfaStylishICPreprocessor import OfaPreprocessorforStylishIC

model_name="damo/ofa_image-caption_coco_distilled_en"
img = "https://shuangqing-public.oss-cn-zhangjiakou.aliyuncs.com/donuts.jpg"

preprocessor=OfaPreprocessorforStylishIC.from_pretrained(model_name, revision="v1.0.1")

stylish_ic=pipeline(Tasks.image_captioning, 
                    model=model_name, 
                    model_revision="v1.0.1",
                    preprocessor=preprocessor)

result=stylish_ic({"image":img, "style":"romantic"})

