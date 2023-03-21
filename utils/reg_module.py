from modelscope.metrics.builder import METRICS
from modelscope.utils.registry import default_group
from metric.stylish_ic_metric import ImageCaptionMetric

# 之后注册模块都在该文件写, 除了preprocessor是用注解注册的
# 注册模块
METRICS.register_module(group_key=default_group, 
                        module_name="image-caption-metric", 
                        module_cls=ImageCaptionMetric)