# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import paddle
import numpy as np
import typing
from pathlib import Path
import logging
import copy
import itertools

from evaluation.evaluator import DatasetEvaluator
from utils import comm


logger = logging.getLogger("ufo.segmentation_evaluator")


class SegEvaluator(object):
    """
    SegEvaluator
    """
    def __init__(self, mode='train'):
        """init
        """
        self.mode = mode

    def reset(self):
        """reset
        """
        pass

    def process(self, inputs, outputs):
        """process
        """
        pass
        
    def evaluate(self):
        """evaluate
        """
        pass


class SegEvaluatorInfer(object):
    """
    SegEvaluatorInfer
    """
    def __init__(self, mode='test', save_path='./'):
        """init
        """
        self.mode = mode
        self.save_path = save_path

    def reset(self):
        """reset
        """
        pass

    def process(self, inputs, outputs):
        """process
        """
        pass
        
    def evaluate(self):
        """evaluate
        """
        pass