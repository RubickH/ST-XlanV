import os
import torch
from torchvision import transforms
from lib.config import cfg
from datasets.coco_dataset import CocoDataset
from datasets.test_dataset import TestDataset
import samplers.distributed
import numpy as np


def sample_collate(batch):
    indices, input_ids_all, input_seq, target_seq, s3d_feats, res152_feats, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_act, aligned_con  = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_ids_all = torch.cat([torch.from_numpy(b) for b in input_ids_all], 0)
    input_seq = torch.cat([torch.from_numpy(b) for b in input_seq], 0)
    target_seq = torch.cat([torch.from_numpy(b) for b in target_seq], 0)

    input_masks_all = torch.cat([torch.from_numpy(b) for b in input_masks_all], 0)
    token_labels_all = torch.cat([torch.from_numpy(b) for b in token_labels_all], 0)
    
    #tag_ids = torch.from_numpy(tag_ids)
    #tag_mask = torch.from_numpy(tag_mask)
    
    atts_num = [x.shape[0] for x in res152_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, res152_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:res152_feats[i].shape[0], :] = res152_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    res152_feats = torch.cat(feat_arr, 0)
    res152_mask = torch.cat(mask_arr, 0)
    
    
    
    atts_num = [x.shape[0] for x in s3d_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, s3d_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:s3d_feats[i].shape[0], :] = s3d_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    s3d_feats = torch.cat(feat_arr, 0)
    s3d_mask = torch.cat(mask_arr, 0)    
    
        
    return indices, input_ids_all, input_seq, target_seq, s3d_feats,s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_act, aligned_con

def sample_collate_val(batch):
    indices, input_ids_all, input_seq, target_seq, s3d_feats, res152_feats, tag_ids, tag_mask, masked_token_ids_all, token_labels_all, aligned_act, aligned_con = zip(*batch)
    
    indices = np.stack(indices, axis=0).reshape(-1)
    input_ids_all = torch.cat([torch.from_numpy(b) for b in input_ids_all], 0)
    atts_num = [x.shape[0] for x in res152_feats]
    max_att_num = np.max(atts_num)

    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, res152_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:res152_feats[i].shape[0], :] = res152_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    res152_feats = torch.cat(feat_arr, 0)
    res152_mask = torch.cat(mask_arr, 0)
    
    
    feat_arr = []
    mask_arr = []
    for i, num in enumerate(atts_num):
        tmp_feat = np.zeros((1, max_att_num, s3d_feats[i].shape[1]), dtype=np.float32)
        tmp_feat[:, 0:s3d_feats[i].shape[0], :] = s3d_feats[i]
        feat_arr.append(torch.from_numpy(tmp_feat))

        tmp_mask = np.zeros((1, max_att_num), dtype=np.float32)
        tmp_mask[:, 0:num] = 1
        mask_arr.append(torch.from_numpy(tmp_mask))

    s3d_feats = torch.cat(feat_arr, 0)
    s3d_mask = torch.cat(mask_arr, 0) 
    
    return indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, aligned_act, aligned_con


def load_train(distributed, epoch, coco_set):
    sampler = samplers.distributed.DistributedSampler(coco_set, epoch=epoch) \
        if distributed else None
    shuffle = cfg.DATA_LOADER.SHUFFLE if sampler is None else False
    
    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TRAIN.BATCH_SIZE,
        shuffle = shuffle, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = cfg.DATA_LOADER.DROP_LAST, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY,
        sampler = sampler, 
        collate_fn = sample_collate
    )
    return loader

def load_val(image_ids_path, s3d_feats_path, s3d_logits_path, res152_feats_path, res152_logits_path, test=0):
    if(test == 0):
        coco_set = CocoDataset(
            image_ids_path = image_ids_path, 
            input_seq = None,  
            s3d_feats_path = s3d_feats_path, 
            s3d_logits_path = s3d_logits_path,
            res152_feats_path = res152_feats_path,
            res152_logits_path = res152_logits_path,
            seq_per_img = 1, 
            max_feats_num = cfg.DATA_LOADER.MAX_FEAT
        )
    else:
        coco_set = TestDataset(
            image_ids_path = image_ids_path, 
            input_seq = None,  
            s3d_feats_path = s3d_feats_path, 
            s3d_logits_path = s3d_logits_path,
            res152_feats_path = res152_feats_path,
            res152_logits_path = res152_logits_path,
            seq_per_img = 1, 
            max_feats_num = cfg.DATA_LOADER.MAX_FEAT
        )

    loader = torch.utils.data.DataLoader(
        coco_set, 
        batch_size = cfg.TEST.BATCH_SIZE,
        shuffle = False, 
        num_workers = cfg.DATA_LOADER.NUM_WORKERS, 
        drop_last = False, 
        pin_memory = cfg.DATA_LOADER.PIN_MEMORY, 
        collate_fn = sample_collate_val
    )
    return loader
