from __future__ import print_function

__author__ = 'tylin'
__version__ = '1.0.1'
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
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load result file and create result api object.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      Version 1.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]
from builtins import int

import json
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import copy

class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = []
        self.cats = []
        if not annotation_file == None:
            print('loading annotations into memory...')
            time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, 'r'))
            print(datetime.datetime.utcnow() - time_t)
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
        anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']] += [ann]
            anns[ann['id']] = ann

        imgs = {im['id']: {} for im in self.dataset['images']}
        for img in self.dataset['images']:
            imgs[img['id']] = img

        cats = []
        catToImgs = []
        #if self.dataset['type'] == 'instances':
        #    cats = {cat['id']: [] for cat in self.dataset['categories']}
        #    for cat in self.dataset['categories']:
        #        cats[cat['id']] = cat
        #    catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
        #    for ann in self.dataset['annotations']:
        #        catToImgs[ann['category_id']] += [ann['image_id']]

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.datset['info'].items():
            print('%s: %s'%(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns],[])
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        if False:#self.dataset['type'] == 'instances':
            if not iscrowd == None:
                ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
            else:
                ids = [ann['id'] for ann in anns]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

   
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for catId in catIds:
                if len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id_] for id_ in ids]
        elif isinstance(ids, int):
            return [self.anns[ids]]


    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id_] for id_ in ids]
        elif isinstance(ids, int):
            return [self.imgs[ids]]


    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset['images'] = [img for img in self.dataset['images']]
        #res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        #res.dataset['type'] = copy.deepcopy(self.dataset['type'])
        #res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print('Loading and preparing results...     ')
        time_t = datetime.datetime.utcnow()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]

        #tmp
        #a = list(set(annsImgIds))
        #with open('../annids.txt','w') as f:
            #for i in range(len(a)):
                #f.write(str(a[i])+'\n')
                
        
        #a = list(set(annsImgIds) & set(self.getImgIds()))
        #with open('../annids_get.txt','w') as f:
            #for i in range(len(a)):
                #f.write(str(a[i])+'\n')
                        
        #assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               #'Results do not correspond to current coco set'
        
        if 'caption' in anns[0]:
            imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
            res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
            for id_, ann in enumerate(anns):
                ann['id'] = id_
        
        print('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset['annotations'] = anns
        res.createIndex()
        return res


    @staticmethod
    def decodeMask(R):
        """
        Decode binary mask M encoded via run-length encoding.
        :param   R (object RLE)    : run-length encoding of binary mask
        :return: M (bool 2D array) : decoded binary mask
        """
        N = len(R['counts'])
        M = np.zeros( (R['size'][0]*R['size'][1], ))
        n = 0
        val = 1
        for pos in range(N):
            val = not val
            for c in range(R['counts'][pos]):
                R['counts'][pos]
                M[n] = val
                n += 1
        return M.reshape((R['size']), order='F')

    @staticmethod
    def encodeMask(M):
        """
        Encode binary mask M using run-length encoding.
        :param   M (bool 2D array)  : binary mask to encode
        :return: R (object RLE)     : run-length encoding of binary mask
        """
        [h, w] = M.shape
        M = M.flatten(order='F')
        N = len(M)
        counts_list = []
        pos = 0
        # counts
        counts_list.append(1)
        diffs = np.logical_xor(M[0:N-1], M[1:N])
        for diff in diffs:
            if diff:
                pos +=1
                counts_list.append(1)
            else:
                counts_list[pos] += 1
        # if array starts from 1. start with 0 counts for 0
        if M[0] == 1:
            counts_list = [0] + counts_list
        return {'size':      [h, w],
               'counts':    counts_list ,
               }

