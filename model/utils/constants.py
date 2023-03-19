import os
import argparse
import json
from .train_conf import *
from functools import partial
from typing import Callable, List, Dict, Union
from modelscope.utils.constant import ConfigKeys, ModeKeys
from modelscope.utils.hub import snapshot_download