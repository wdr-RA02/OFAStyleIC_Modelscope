import os
import argparse
import json
from utils.train_conf import *
from functools import partial
from typing import Callable, List, Dict, Union, Any
from modelscope.utils.constant import ConfigKeys, ModeKeys, Tasks
from modelscope.utils.hub import snapshot_download
from modelscope.outputs import OutputKeys
from .collate_fn_multitask import collate_mul_tasks, build_collator