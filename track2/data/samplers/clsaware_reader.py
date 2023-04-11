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

import os
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
from paddle.io import DistributedBatchSampler
    
from ppdet.utils.logger import setup_logger
from collections import Counter

logger = setup_logger('reader')
MAIN_PID = os.getpid()


class VehicleMultiTaskClassAwareSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        super(VehicleMultiTaskClassAwareSampler, self).__init__(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size
        self.category_imgids = self._classaware_sampler(dataset.img_items)

        # counter = [0 for _ in range(len(self.category_imgids))]
        # for i in range(len(self.category_imgids)):
        #     counter += len(self.category_imgids[i])
        # self.class_sampler_prob = np.array(counter) / sum(counter)
        self.class_sampler_prob = [1.0/len(self.category_imgids) for _ in range(len(self.category_imgids))]
        
    def __iter__(self):

        while True:
            batch_index = []
            random_categories = list(np.random.choice(list(range(len(self.category_imgids))), 
                                                    self.batch_size, replace=True, 
                                                    p=self.class_sampler_prob))
            for cls, count in Counter(random_categories).items(): 
                cur_ids = list(np.random.choice(self.category_imgids[cls], count, replace=False))
                batch_index.extend(cur_ids)
            if self.shuffle:
                np.random.RandomState(self.epoch).shuffle(batch_index)
                self.epoch += 1

            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index

    def _classaware_sampler(self, roidbs):
        category_imgids = {}
        for i, roidb in enumerate(roidbs):
            label = roidb[1] # label
            if label not in category_imgids:
                category_imgids[label] = []
            category_imgids[label].append(i)  # 每个类别对应的图片id

        return category_imgids

