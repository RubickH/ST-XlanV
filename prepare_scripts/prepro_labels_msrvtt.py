"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.lua

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/images is (N,3,256,256) uint8 array of raw image data in RGB format
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image


def encode_captions(imgs, params):
    """ 
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed 
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(imgs[key]['captions']) for key in imgs.keys()) # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    
    
    
    
    for i in range(N):
        key = 'video'+str(i)
        n = len(imgs[key]['captions'])
        assert n > 0, 'error: some image has no captions'

        Li = []
        for j,s in enumerate(imgs[key]['captions']):
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            label_arrays.append(s)


        # note: word indices are 1-indexed, and captions are padded with zeros
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    #L = np.concatenate(label_arrays, axis=0) # put all the labels together
    assert len(label_arrays) == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', len(label_arrays))
    return label_arrays, label_start_ix, label_end_ix, label_length

def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    #imgs = imgs['images']

    #seed(123) # make reproducible

    # create the vocab
    #vocab, counts = build_vocab(imgs, params)
    #itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    #wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    # encode captions in large arrays, ready to ship to hdf5 file
    label_arrays, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params)

    # create output h5 file
    N = len(imgs)
    json_dict = {'labels': label_arrays}
    json.dump(json_dict, open(params['output_json'],'w'))
    
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()    
    
  

    # create output json file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default = '/media/hyq/part2/ACMMM2021/XLanV/data_common/caption_msrvtt.json', help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='/media/hyq/part2/ACMMM2021/XLanV/data_common/msrvtt.json', help='output json file')
    parser.add_argument('--output_h5', default='/media/hyq/part2/ACMMM2021/XLanV/data_common/msrvtt', help='output h5 file')
    # options
    parser.add_argument('--max_length', default=20, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params)
