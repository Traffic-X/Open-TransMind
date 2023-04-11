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

from evaluation.coco_utils import get_infer_results, cocoapi_eval
from evaluation.evaluator import DatasetEvaluator
from utils import comm


logger = logging.getLogger("ufo.cocodet_evaluator")


class CocoDetEvaluatorSingleTask(DatasetEvaluator):
    """CocoDetEvaluatorSingleTask
    """
    def __init__(self, anno_file='', clsid2catid={}, classwise=False, output_eval=None,
        bias=0, IouType='bbox', save_prediction_only=False, **kwargs):
        self.anno_file = anno_file
        self.clsid2catid = clsid2catid
        self.classwise = classwise
        self.output_eval = output_eval
        self.bias = bias
        self.save_prediction_only = save_prediction_only
        self.iou_type = IouType
        self.parallel_evaluator = kwargs.get('parallel_evaluator', True)
        self.num_valid_samples = kwargs.get('num_valid_samples', None)

        if self.output_eval is not None:
            Path(self.output_eval).mkdir(exist_ok=True)

        self.reset()

    def reset(self):
        """reset
        """
        self.results = []
        self.infer_results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def process(self, inputs, outputs):
        """process
        """
        # remove task dict
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'

        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        # multi-scale inputs: all inputs have same im_id
        if isinstance(inputs, typing.Sequence):
            im_id = inputs[0]['im_id']
        else:
            im_id = inputs['im_id']
        outputs['im_id'] = im_id
        
        for k, v in outputs.items():
            outputs[k] = v.cpu()
        self.results.append(outputs)
    
    def evaluate(self):
        """evaluate
        """
        if self.parallel_evaluator and  comm.get_world_size() > 1:
            comm.synchronize()
            results = comm.gather(self.results)
            results = list(itertools.chain(*results))
            if not comm.is_main_process():
                return {}
        else:
            results = self.results
        if self.num_valid_samples is not None:
            self.results = self.results[:self.num_valid_samples]
        if not comm.is_main_process():
                return {}
        for result in results:
            for k, v in result.items():
                result[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

            infer_result = get_infer_results(result, self.clsid2catid, bias=self.bias)
            self.infer_results['bbox'] += infer_result['bbox'] if 'bbox' in infer_result else []
            self.infer_results['mask'] += infer_result['mask'] if 'mask' in infer_result else []
            self.infer_results['segm'] += infer_result['segm'] if 'segm' in infer_result else []
            self.infer_results['keypoint'] += infer_result['keypoint'] if 'keypoint' in infer_result else []
        
        if len(self.infer_results['bbox']) > 0:
            output = "bbox.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.infer_results['bbox'], f)
                logger.info('The bbox result is saved to bbox.json.')

            if self.save_prediction_only:
                logger.info('The bbox result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                bbox_stats = cocoapi_eval(
                    output,
                    'bbox',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['bbox'] = bbox_stats
                sys.stdout.flush()

        if len(self.infer_results['mask']) > 0:
            output = "mask.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.infer_results['mask'], f)
                logger.info('The mask result is saved to mask.json.')

            if self.save_prediction_only:
                logger.info('The mask result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.infer_results['segm']) > 0:
            output = "segm.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.infer_results['segm'], f)
                logger.info('The segm result is saved to segm.json.')

            if self.save_prediction_only:
                logger.info('The segm result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                seg_stats = cocoapi_eval(
                    output,
                    'segm',
                    anno_file=self.anno_file,
                    classwise=self.classwise)
                self.eval_results['mask'] = seg_stats
                sys.stdout.flush()

        if len(self.infer_results['keypoint']) > 0:
            output = "keypoint.json"
            if self.output_eval:
                output = os.path.join(self.output_eval, output)
            with open(output, 'w') as f:
                json.dump(self.infer_results['keypoint'], f)
                logger.info('The keypoint result is saved to keypoint.json.')

            if self.save_prediction_only:
                logger.info('The keypoint result is saved to {} and do not '
                            'evaluate the mAP.'.format(output))
            else:
                style = 'keypoints'
                use_area = True
                if self.iou_type == 'keypoints_crowd':
                    style = 'keypoints_crowd'
                    use_area = False
                keypoint_stats = cocoapi_eval(
                    output,
                    style,
                    anno_file=self.anno_file,
                    classwise=self.classwise,
                    use_area=use_area)
                self.eval_results['keypoint'] = keypoint_stats
                sys.stdout.flush()
        
        eval_results = {}
        eval_results['precision_avg_all_100'] = self.eval_results['bbox'][0]
        eval_results['precision_0.50_all_100'] = self.eval_results['bbox'][1]
        eval_results['precision_0.75_all_100'] = self.eval_results['bbox'][2]
        eval_results['precision_avg_small_100'] = self.eval_results['bbox'][3]
        eval_results['precision_avg_medium_100'] = self.eval_results['bbox'][4]
        eval_results['precision_avg_large_100'] = self.eval_results['bbox'][5]
        eval_results['recall_avg_all_1'] = self.eval_results['bbox'][6]
        eval_results['recall_avg_all_10'] = self.eval_results['bbox'][7]
        eval_results['recall_avg_all_100'] = self.eval_results['bbox'][8]
        eval_results['recall_avg_small_100'] = self.eval_results['bbox'][9]
        eval_results['recall_avg_medium_100'] = self.eval_results['bbox'][10]
        eval_results['recall_avg_large_100'] = self.eval_results['bbox'][11]

        return eval_results


class CocoDetEvaluatorSingleTaskInfer(DatasetEvaluator):
    """CocoDetEvaluatorSingleTaskInfer
    """
    def __init__(self, anno_file='', clsid2catid={}, classwise=False, output_eval=None,
            bias=0, IouType='bbox', save_prediction_only=False, **kwargs):
        self.anno_file = anno_file
        self.clsid2catid = clsid2catid
        self.classwise = classwise
        self.output_eval = output_eval
        self.bias = bias
        self.save_prediction_only = save_prediction_only
        self.iou_type = IouType
        self.parallel_evaluator = kwargs.get('parallel_evaluator', True)
        self.num_valid_samples = kwargs.get('num_valid_samples', None)

        # if self.output_eval is not None:
        #     Path(self.output_eval).mkdir(exist_ok=True)

        self.reset()

    def reset(self):
        """reset
        """
        self.results = []
        self.infer_results = {'bbox': [], 'mask': [], 'segm': [], 'keypoint': []}
        self.eval_results = {}

    def process(self, inputs, outputs):
        """process
        """
        # remove task dict
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'

        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        # multi-scale inputs: all inputs have same im_id
        if isinstance(inputs, typing.Sequence):
            im_id = inputs[0]['im_id']
        else:
            im_id = inputs['im_id']
        outputs['im_id'] = im_id
        
        for k, v in outputs.items():
            outputs[k] = v.cpu()
        self.results.append(outputs)
    
    def evaluate(self):
        """evaluate
        """
        if self.parallel_evaluator and  comm.get_world_size() > 1:
            comm.synchronize()
            results = comm.gather(self.results)
            results = list(itertools.chain(*results))
            if not comm.is_main_process():
                return {}
        else:
            results = self.results
        if self.num_valid_samples is not None:
            self.results = self.results[:self.num_valid_samples]
        if not comm.is_main_process():
            return {}
        for result in results:
            for k, v in result.items():
                result[k] = v.numpy() if isinstance(v, paddle.Tensor) else v

            infer_result = get_infer_results(result, self.clsid2catid, bias=self.bias)
            self.infer_results['bbox'] += infer_result['bbox'] if 'bbox' in infer_result else []
    
        return {'dec': self.infer_results['bbox']}