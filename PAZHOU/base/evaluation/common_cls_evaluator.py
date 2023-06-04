# encoding: utf-8

import copy
import itertools
import json
import logging
from collections import OrderedDict
import os

import paddle

from utils import comm
from evaluation.evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)

# eval mode for training
class CommonClasEvaluatorSingleTask(DatasetEvaluator):
    """CommonClasEvaluatorSingleTask
    """
    def __init__(self, cfg, output_dir=None, num_valid_samples=None, **kwargs):
        self.cfg = cfg
        self._output_dir = output_dir
        self.task_type = kwargs.get('task_type', 'brand')

        self._predictions = []
        self.topk = (1,)
        self._num_valid_samples = num_valid_samples

        self.num_classes = kwargs.get('num_classes', 3)

    def reset(self):
        """reset
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """process
        """
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'
        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        pred_logits = outputs
        labels = inputs["targets"]
        with paddle.no_grad():
            maxk = max(self.topk)
            batch_size = labels.shape[0]
            for i in range(batch_size):
                label = labels[i]
                result = -paddle.ones((2,))
                _, pred = pred_logits[i].topk(maxk, -1, True, True)
                result[0] = int(label)
                result[1] = int(pred)
                self._predictions.append(result)

    def evaluate(self):
        """evaluate
        """
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process(): return {}
        else:
            predictions = self._predictions
        
        conf_mat = paddle.zeros((self.num_classes, self.num_classes))
        correct = 0
        total_num = 0
        for prediction in predictions:
            label = int(prediction[0])
            if label != -1:
                pred = int(prediction[1])
                if label == pred:
                    correct += 1
                total_num += 1
                conf_mat[label, pred] += 1
        
        self._results = OrderedDict()
        self._results["Acc@1"] = correct / total_num
        
        return copy.deepcopy(self._results)


# infer mode only for test dataset
class CommonClasEvaluatorSingleTaskInfer(DatasetEvaluator):
    """CommonClasEvaluatorSingleTaskInfer
    """
    def __init__(self, cfg, output_dir=None, num_valid_samples=None, **kwargs):
        self.cfg = cfg
        self._output_dir = output_dir
        self.task_type = kwargs.get('task_type', 'brand')

        self._predictions = []
        self.topk = (1,)
        self._num_valid_samples = num_valid_samples

        self.num_classes = kwargs.get('num_classes', 3)

    def reset(self):
        """reset
        """
        self._predictions = []

    def process(self, inputs, outputs):
        """process
        """
        assert len(inputs) == 1, 'support only single task evaluation'
        assert len(outputs) == 1, 'support only single task evaluation'
        inputs = list(inputs.values())[0]
        outputs = list(outputs.values())[0]

        pred_logits = outputs
        im_id = inputs["im_id"]
        self.id2imgname = inputs["id2imgname"]
        batch_size = im_id.shape[0]
        with paddle.no_grad():
            maxk = max(self.topk)
            for i in range(batch_size):
                result = -paddle.ones((2,))
                _, pred = pred_logits[i].topk(maxk, -1, True, True)
                result[0] = int(pred)
                result[1] = int(im_id[i])
                self._predictions.append(result)

    def evaluate(self):
        """evaluate
        """
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process(): return {}
        else:
            predictions = self._predictions
        
        pred_res = []
        for prediction in predictions:
            img_path = self.id2imgname[int(prediction[1])]
            pred = int(prediction[0])
            tmp = dict()
            tmp[os.path.basename(img_path[0])] = pred
            pred_res.append(tmp)
            
        return {'cls': pred_res}