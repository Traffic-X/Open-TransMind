# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
# from torchvision.datasets import VisionDataset

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def hdf5_bunch_get(hdf5_dataset, indices, sorted=False):
    if sorted:
        items = [p for p in hdf5_dataset[indices]]
        items = np.array(items)
    else:
        img_idxs_sorted, reverse_indices = np.unique(indices, return_inverse=True)
        items = [p for p in hdf5_dataset[img_idxs_sorted.tolist()]]
        items = np.array(items)[reverse_indices]
    return items


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        # self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        # self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.catToImgs = defaultdict(list)
        self.catToImgsSet = None
        self.num_categ = None
        self.num_image = None
        self.num_annos = None

        if not annotation_file == None:
            self.anno_file = annotation_file
            self.anno_hdf5 = None
            print('loading annotations into memory...')
            tic = time.time()
            # dataset = json.load(open(annotation_file, 'r'))
            # assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            # self.dataset = dataset
            self.createIndex()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def open_hdf5(self):
        # we need this function to open anno_hdf5 file after the object is forked
        if self.anno_hdf5 is None:
            self.anno_hdf5 = h5py.File(self.anno_file, 'r')

    def createIndex(self):
        # create index
        print('creating index...')

        # don't set self.anno_hdf5 in this function since h5py file can't be forked
        # in multiple processes
        anno_hdf5 = h5py.File(self.anno_file, 'r')
        self.num_categ = anno_hdf5['name'].shape[0]
        self.num_image = anno_hdf5['file_name'].shape[0]
        self.num_annos = anno_hdf5['image_id'].shape[0]

        # build catToImgs: category id => [image_id0 image_id1]
        catToImgs = defaultdict(list)

        # get all items will be much faster compared to access one by one
        category_ids = anno_hdf5['category_id'][:].tolist()
        image_ids = anno_hdf5['image_id'][:].tolist()
        for idx, cat_id in enumerate(category_ids):
            image_id = image_ids[idx]
            catToImgs[cat_id].append(image_id)
        self.catToImgs = catToImgs  # category id => [image_id0 image_id1]

        catToImgsSet = {}
        for key, value in catToImgs.items():
            catToImgsSet[key] = set(value)
        self.catToImgsSet = catToImgsSet

        # build cats
        cat_ids = anno_hdf5['id'][:].tolist()
        cat_names = anno_hdf5['name'][:].tolist()
        cat_supercategories = anno_hdf5['supercategory'][:].tolist()
        cats = {}
        for cid, name, supercategory in zip(cat_ids, cat_names, cat_supercategories):
            cats[cid] = {'id': cid, 'name': name, 'supercategory': supercategory}
        self.cats = cats

        # anns, cats, imgs = {}, {}, {}
        # imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        # if 'annotations' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         imgToAnns[ann['image_id']].append(ann)
        #         anns[ann['id']] = ann
        #
        # if 'images' in self.dataset:
        #     for img in self.dataset['images']:
        #         imgs[img['id']] = img
        #
        # if 'categories' in self.dataset:
        #     for cat in self.dataset['categories']:
        #         cats[cat['id']] = cat
        #
        # if 'annotations' in self.dataset and 'categories' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         catToImgs[ann['category_id']].append(ann['image_id'])
        # create class members
        # self.anns = anns  # anno id => anno
        # self.imgToAnns = imgToAnns  # image id => [ann0 anno1 anno2]
        # self.catToImgs = catToImgs  # category id => [image_id0 image_id1]
        # self.imgs = imgs  # image id => image_info
        # self.cats = cats  # cat id => cat info

        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        raise NotImplementedError
        # for key, value in self.dataset['info'].items():
        #     print('{}: {}'.format(key, value))

    # needed by v2x
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        self.open_hdf5()

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anno_ids = range(self.num_annos)
            # anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anno_ids = []
                for imgId in imgIds:
                    if imgId < self.num_image:
                        start_idx = self.anno_hdf5['start_idx'][imgId]
                        end_idx = self.anno_hdf5['start_idx'][imgId + 1]
                        for _id in range(start_idx, end_idx):
                            anno_ids.append(_id)
                # lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                # anns = list(itertools.chain.from_iterable(lists))
            else:
                anno_ids = range(self.num_annos)
                # anns = self.dataset['annotations']

            if len(catIds) != 0:
                cate_ids = hdf5_bunch_get(self.anno_hdf5['category_id'], anno_ids)
                anno_ids = [_id for _id, cid in zip(anno_ids, cate_ids) if cid in catIds]

            if len(areaRng) != 0:
                areas = hdf5_bunch_get(self.anno_hdf5['area'], anno_ids)
                anno_ids = [_id for _id, area in zip(anno_ids, areas) if area > areaRng[0] and area < areaRng[1]]

            # anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            # anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

        if not iscrowd == None:
            iscrowds = hdf5_bunch_get(self.anno_hdf5['iscrowd'], anno_ids)
            anno_ids = [id_ for id_, iscrowd_ in zip(anno_ids, iscrowds) if iscrowd_ == iscrowd]
            # ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            anno_ids = anno_ids
            # ids = [ann['id'] for ann in anns]
        return anno_ids

    # needed by v2x
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        self.open_hdf5()

        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats_ids = self.anno_hdf5['id'][:].tolist()
            # cats = self.dataset['categories']
        else:
            cats_ids = self.anno_hdf5['id'][:].tolist()
            # cats = self.dataset['categories']
            if len(catNms) != 0:
                cats_ids = [id_ for id_ in cats_ids if self.cats[id_]['name'] in catNms]

            if len(supNms) != 0:
                cats_ids = [id_ for id_ in cats_ids if self.cats[id_]['supercategory'] in supNms]

        # ids = [cat['id'] for cat in cats]
        return cats_ids

    # needed by v2x
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        self.open_hdf5()

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = range(self.num_image)
            # ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = self.catToImgsSet[catId]
                else:
                    ids &= self.catToImgsSet[catId]
        return list(ids)

    def process_attr(self, key, value):
        if key in ['file_name', 'name', 'supercategory', 'seg_direction', 'seg_sub_type']:
            value = value.decode()
        elif key in ['bbox', 'lights', 'vis_line']:
            value = value.tolist()
        elif key == 'mid_line':
            a, b = value.tolist()
            a = int(a)  # the first value of mid_line is int
            value = [a, b]
        elif key == 'box_3d':
            value = value.tolist()
            new_value = []
            for idx in range(len(value) // 2):
                new_value.append({'y': value[2 * idx], 'x': value[2 * idx + 1]})
            value = new_value
        elif key in ['rotation_ori', 'rotation_3d', 'rotation_flip']:
            value = value.tolist()
            value = {'theta': value[0], 'phi': value[1], 'psi': value[2]}
        elif key in ['center_3d']:
            value = value.tolist()
            value = {'y': value[0], 'x': value[1], 'z': value[2]}
        elif key in ['transform']:
            value = value.tolist()
            value = {
                'rotation': {'y': value[0], 'x': value[1], 'z': value[2], 'w': value[3]},
                'translation': {'y': value[4], 'x': value[5], 'z': value[6]}
            }

        return value

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        self.open_hdf5()

        if type(ids) == int:
            ids = [ids]
        if len(ids) == 0:
            return []

        # this op should be fast, since it will be frequently invoked
        attr_pos = hdf5_bunch_get(self.anno_hdf5['attr_pos'], ids)

        annos = [{} for _ in range(len(ids))]
        for idx, key in enumerate(self.anno_hdf5.attrs['anno_keys']):
            one_attr_pos = attr_pos[:, idx]
            # remove negative index
            one_attr_pos_exist = one_attr_pos[one_attr_pos >= 0]
            one_attr_exist = hdf5_bunch_get(self.anno_hdf5[key], one_attr_pos_exist)
            sel_idx = 0
            for ann_idx, pos in enumerate(one_attr_pos):
                if pos >= 0:
                    annos[ann_idx][key] = self.process_attr(key, one_attr_exist[sel_idx])
                    sel_idx += 1
                else:
                    if key in ['track_id', 'ground_point', 'line_angle']:
                        # for these attributes, it is either int or not in the anno
                        # it is ugly, but this enables minimum changes to v2x.py
                        continue
                    annos[ann_idx][key] = 'None'

        # if _isArrayLike(ids):
        #     return [self.anns[id] for id in ids]
        # elif type(ids) == int:
        #     return [self.anns[ids]]
        return annos

    # needed by v2x
    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        self.open_hdf5()

        # if type(ids) == int:
        #     ids = [ids]
        # cats = [{} for _ in range(len(ids))]
        # cat_keys = ['supercategory', 'name']
        # for idx, key in enumerate(cat_keys):
        #     one_attr = hdf5_bunch_get(self.anno_hdf5[key], ids)
        #     for cat_idx, id_ in enumerate(ids):
        #         cats[cat_idx][key] = self.process_attr(key, one_attr[cat_idx])
        # return cats

        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    # needed by v2x
    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        self.open_hdf5()

        if type(ids) == int:
            ids = [ids]
        imgs = [{} for _ in range(len(ids))]
        img_keys = ['file_name', 'width', 'height']
        for idx, key in enumerate(img_keys):
            one_attr = hdf5_bunch_get(self.anno_hdf5[key], ids)
            for img_idx, id_ in enumerate(ids):
                imgs[img_idx][key] = self.process_attr(key, one_attr[img_idx])

        # if _isArrayLike(ids):
        #     return [self.imgs[id] for id in ids]
        # elif type(ids) == int:
        #     return [self.imgs[ids]]
        return imgs

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        raise NotImplementedError
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton']) - 1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(x[v > 0], y[v > 0], 'o', markersize=8, markerfacecolor=c, markeredgecolor='k',
                             markeredgewidth=2)
                    plt.plot(x[v > 1], y[v > 1], 'o', markersize=8, markerfacecolor=c, markeredgecolor=c,
                             markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                            [bbox_x + bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        raise NotImplementedError

        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        tic = time.time()
        if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
            'Results do not correspond to current coco set'
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id, ann in enumerate(anns):
                ann['id'] = id + 1
        elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if not 'segmentation' in ann:
                    ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'segmentation' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann['area'] = maskUtils.area(ann['segmentation'])
                if not 'bbox' in ann:
                    ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                ann['id'] = id + 1
                ann['iscrowd'] = 0
        elif 'keypoints' in anns[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(anns):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1 - x0) * (y1 - y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        print('DONE (t={:0.2f}s)'.format(time.time() - tic))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        '''
        Download COCO images from mscoco.org server.
        :param tarDir (str): COCO results directory name
               imgIds (list): images to be downloaded
        :return:
        '''
        raise NotImplementedError
        if tarDir is None:
            print('Please specify target directory')
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img['file_name'])
            if not os.path.exists(fname):
                urlretrieve(img['coco_url'], fname)
            print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time() - tic))

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert (type(data) == np.ndarray)
        print(data.shape)
        assert (data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i, N))
            ann += [{
                'image_id': int(data[i, 0]),
                'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                'score': data[i, 5],
                'category_id': int(data[i, 6]),
            }]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        raise NotImplementedError
        t = self.imgs[ann['image_id']]
        h, w = t['height'], t['width']
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        raise NotImplementedError
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m


# class CocoDetectionHDF(VisionDataset):
#     """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         annFile (string): Path to json annotation file.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.ToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         transforms (callable, optional): A function/transform that takes input sample and its target as entry
#             and returns a transformed version.
#     """

#     def __init__(
#             self,
#             root: str,
#             annFile: str,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             transforms: Optional[Callable] = None,
#     ):
#         super().__init__(root, transforms, transform, target_transform)
#         # from pycocotools.coco import COCO

#         self.coco = COCO(annFile)
#         self.ids = list(range(self.coco.num_image))

#     def _load_image(self, id: int) -> Image.Image:
#         path = self.coco.loadImgs(id)[0]["file_name"]
#         return Image.open(os.path.join(self.root, path)).convert("RGB")

#     def _load_target(self, id) -> List[Any]:
#         return self.coco.loadAnns(self.coco.getAnnIds(id))

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         id = self.ids[index]
#         image = self._load_image(id)
#         target = self._load_target(id)

#         if self.transforms is not None:
#             image, target = self.transforms(image, target)

#         return image, target

#     def __len__(self) -> int:
#         return len(self.ids)
