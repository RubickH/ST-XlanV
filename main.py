import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import models
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file
import pdb
class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        if cfg.SEED > 0:
            random.seed(cfg.SEED)
            torch.manual_seed(cfg.SEED)
            torch.cuda.manual_seed_all(cfg.SEED)

        self.num_gpus = torch.cuda.device_count()
        self.distributed = self.num_gpus > 1
        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://"
            )
        self.device = torch.device("cuda")
        self.pre_train = args.pre_train
        self.rl_stage = False
        self.setup_logging()
        if args.use_action:
            self.setup_dataset2()
        else:
            self.setup_dataset()
        self.setup_network()
        self.val_evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            s3d_feats = cfg.DATA_LOADER.VAL_S3D_FEATS,
            s3d_logits = cfg.DATA_LOADER.VAL_S3D_LOGITS,
            res152_feats = cfg.DATA_LOADER.VAL_RES152_FEATS,
            res152_logits = cfg.DATA_LOADER.VAL_RES152_LOGITS,            
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
        )
        
        self.scorer = Scorer()

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        if self.distributed and dist.get_rank() > 0:
            return
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info('Training with config:')
        self.logger.info(pprint.pformat(cfg))

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE, cfg.MODEL.IF_POS, cfg.MODEL.IF_TAG)

        if self.distributed:
            # this should be removed if we update BatchNorm stats
            self.model = torch.nn.parallel.DistributedDataParallel(
                model.to(self.device), 
                device_ids = [self.args.local_rank], 
                output_device = self.args.local_rank,
                broadcast_buffers = False,
                find_unused_parameters=True
            )
        else:
            self.model = torch.nn.DataParallel(model).to(self.device)

        if self.args.resume >= 0:
            print('Resuming from checkpoint {}!'.format(self.args.resume))
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage), strict=False
            )
            self.start_epoch = self.args.resume
            self.scheduled_sampling(self.start_epoch)
        else:
            self.start_epoch = 0

        self.optim = Optimizer(self.model, self.training_loader)
        self.xe_criterion = losses.create(cfg.LOSSES.XE_TYPE).to(self.device)
        self.rl_criterion = losses.create(cfg.LOSSES.RL_TYPE).to(self.device)
        
    def setup_dataset(self):
        self.coco_set = datasets.coco_dataset.CocoDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID, 
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH, 
            s3d_feats_path = cfg.DATA_LOADER.TRAIN_S3D_FEATS, 
            s3d_logits_path = cfg.DATA_LOADER.TRAIN_S3D_LOGITS,
            res152_feats_path = cfg.DATA_LOADER.TRAIN_RES152_FEATS,
            res152_logits_path = cfg.DATA_LOADER.TRAIN_RES152_LOGITS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feats_num = cfg.DATA_LOADER.MAX_FEAT,
            pretrain=self.pre_train,
        )
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, 0, self.coco_set)

    def setup_dataset2(self):
        from datasets import action_dataset
        self.coco_set = datasets.action_dataset.ActionDataset(
            image_ids_path = cfg.DATA_LOADER.TRAIN_ID,
            input_seq = cfg.DATA_LOADER.INPUT_SEQ_PATH,
            s3d_feats_path = cfg.DATA_LOADER.TRAIN_S3D_FEATS,
            s3d_logits_path = cfg.DATA_LOADER.TRAIN_S3D_LOGITS,
            res152_feats_path = cfg.DATA_LOADER.TRAIN_RES152_FEATS,
            res152_logits_path = cfg.DATA_LOADER.TRAIN_RES152_LOGITS,
            seq_per_img = cfg.DATA_LOADER.SEQ_PER_IMG,
            max_feats_num = cfg.DATA_LOADER.MAX_FEAT,
            pretrain=self.pre_train,
        )
        print(len(self.coco_set))
        print("load Action Dataset!")
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, 0, self.coco_set)

    def setup_loader(self, epoch):
        self.training_loader = datasets.data_loader.load_train(
            self.distributed, epoch, self.coco_set)

    def eval(self, epoch):
        if (epoch + 1) % cfg.SOLVER.TEST_INTERVAL != 0:
            return None
        if self.distributed and dist.get_rank() > 0:
            return None
            
        val_res = self.val_evaler(self.model, 'val_' + str(epoch + 1))
        self.logger.info('######## Epoch (VAL)' + str(epoch + 1) + ' ########')
        self.logger.info(str(val_res))

        with open(os.path.join(args.folder,'eval_results.txt'),'a') as f:
            f.write('Epoch {} \n'.format(epoch+1))
            f.write('validation results:\n')
            f.write(str(val_res)+'\n')
            
        val = 0
        for score_type, weight in zip(cfg.SCORER.TYPES, cfg.SCORER.WEIGHTS):
            val -= val_res[score_type] * weight
        return val

    def snapshot_path(self, name, epoch):
        snapshot_path = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_path, name + "_" + str(epoch) + ".pth")

    def save_model(self, epoch):
        if (epoch + 1) % cfg.SOLVER.SNAPSHOT_ITERS != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        snapshot_path = os.path.join(cfg.ROOT_DIR, 'snapshot')
        if not os.path.exists(snapshot_path):
            os.mkdir(snapshot_path)
        torch.save(self.model.state_dict(), self.snapshot_path("caption_model", epoch+1))

    def make_kwargs(self, indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_act, aligned_con):
        seq_mask = (input_seq > 0).type(torch.LongTensor).to(self.device)
        seq_mask[:,0] += 1
        seq_mask_sum = seq_mask.sum(-1)
        max_len = int(seq_mask_sum.max())
        input_seq = input_seq[:, 0:max_len].contiguous()
        target_seq = target_seq[:, 0:max_len].contiguous()
        tag_ids = tag_ids.contiguous()
        tag_mask = tag_mask.contiguous()
        
        kwargs = {
            cfg.PARAM.INDICES: indices,
            cfg.PARAM.INPUT_IDS: input_ids_all,
            cfg.PARAM.INPUT_SENT: input_seq,
            cfg.PARAM.TARGET_SENT: target_seq,
            cfg.PARAM.S3D_FEATS: s3d_feats,
            cfg.PARAM.S3D_FEATS_MASK: s3d_mask,
            cfg.PARAM.RES152_FEATS: res152_feats,
            cfg.PARAM.RES152_FEATS_MASK: res152_mask,
            cfg.PARAM.TAG_IDS: tag_ids,
            cfg.PARAM.TAG_MASK: tag_mask,
            cfg.PARAM.INPUT_MASK_ALL: input_masks_all,
            cfg.PARAM.TOKEN_LABELS_ALL: token_labels_all,
            cfg.PARAM.ALIGNED_ACT: aligned_act,
            cfg.PARAM.ALIGNED_CON: aligned_con
        }
        return kwargs

    def scheduled_sampling(self, epoch):
        if epoch > cfg.TRAIN.SCHEDULED_SAMPLING.START:
            frac = (epoch - cfg.TRAIN.SCHEDULED_SAMPLING.START) // cfg.TRAIN.SCHEDULED_SAMPLING.INC_EVERY
            ss_prob = min(cfg.TRAIN.SCHEDULED_SAMPLING.INC_PROB * frac, cfg.TRAIN.SCHEDULED_SAMPLING.MAX_PROB)
            self.model.module.ss_prob = ss_prob

    def display(self, epoch, iteration, data_time, batch_time, losses, loss_info, acc_mlm, acc_match):
        if iteration % cfg.SOLVER.DISPLAY != 0:
            return
        if self.distributed and dist.get_rank() > 0:
            return
        info_str = ' (DataTime/BatchTime: {:.3}/{:.3}) losses = {:.5}'.format(data_time.avg, batch_time.avg, losses.avg)
        self.logger.info('Epoch '+str(epoch)+', Iteration ' + str(iteration) + info_str +', lr = ' +  str(self.optim.get_lr()))
        if self.pre_train:
            self.logger.info('  acc_mlm' + ' = ' + str(acc_mlm.avg) + '  acc_match' + ' = ' + str(acc_match.avg))
        else:
            for name in sorted(loss_info):
                self.logger.info('  ' + name + ' = ' + str(loss_info[name]))
        data_time.reset()
        batch_time.reset()
        losses.reset()
        acc_mlm.reset()
        acc_match.reset()


    def forward(self, kwargs):
        if self.pre_train:
            loss, loss_info = self.model.module.forward_pretrain(**kwargs)

        elif self.rl_stage == False:
            logit = self.model(**kwargs)
            loss, loss_info = self.xe_criterion(logit, kwargs[cfg.PARAM.TARGET_SENT])
            
        else:
            ids = kwargs[cfg.PARAM.INDICES]
                        
            # max
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = True         

            self.model.eval()
            with torch.no_grad():
                seq_max, logP_max = self.model.module.decode(**kwargs)
            self.model.train()
            rewards_max, rewards_info_max = self.scorer(ids, seq_max.data.cpu().numpy().tolist())
            rewards_max = utils.expand_numpy(rewards_max)

            ids = utils.expand_numpy(ids)
            

            # sample
            kwargs['BEAM_SIZE'] = 1
            kwargs['GREEDY_DECODE'] = False

            seq_sample, logP_sample = self.model.module.decode(**kwargs)
            rewards_sample, rewards_info_sample = self.scorer(ids, seq_sample.data.cpu().numpy().tolist())
            
            rewards = rewards_sample - rewards_max
            rewards = torch.from_numpy(rewards).float().cuda()
            loss = self.rl_criterion(seq_sample, logP_sample, rewards)
            loss_info = {}
            for key in rewards_info_sample:
                loss_info[key + '_sample'] = rewards_info_sample[key]
            for key in rewards_info_max:
                loss_info[key + '_max'] = rewards_info_max[key]
        return loss, loss_info

    def train(self):
        torch.cuda.empty_cache()
        self.model.train()
        self.optim.zero_grad()
        iteration = 0
        for epoch in range(self.start_epoch, cfg.SOLVER.MAX_EPOCH):
            if epoch >= cfg.TRAIN.REINFORCEMENT.START:
                self.rl_stage = True
            self.setup_loader(epoch)
           
            start = time.time()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            losses = AverageMeter()
            acc_mlm = AverageMeter()
            acc_match = AverageMeter()
            #val = self.eval(epoch)
            
            for _, (indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_act, aligned_con) in enumerate(self.training_loader):

                data_time.update(time.time() - start)
                
                input_ids_all = input_ids_all.to(self.device)
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)
                s3d_feats = s3d_feats.to(self.device)
                s3d_mask = s3d_mask.to(self.device)
                res152_feats = res152_feats.to(self.device)
                res152_mask = res152_mask.to(self.device)
                tag_ids = torch.from_numpy(np.array(tag_ids)).to(self.device)
                tag_mask = torch.from_numpy(np.array(tag_mask)).to(self.device)
                aligned_act = torch.from_numpy(np.array(aligned_act)).to(self.device)
                aligned_con = torch.from_numpy(np.array(aligned_con)).to(self.device)
                input_masks_all = input_masks_all.to(self.device)
                token_labels_all = token_labels_all.to(self.device)
                
                
                kwargs = self.make_kwargs(indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, input_masks_all, token_labels_all, aligned_act, aligned_con)
               
                loss, loss_info = self.forward(kwargs)
                loss.backward()
                utils.clip_gradient(self.optim.optimizer, self.model,
                    cfg.SOLVER.GRAD_CLIP_TYPE, cfg.SOLVER.GRAD_CLIP)
                self.optim.step()
                self.optim.zero_grad()
                self.optim.scheduler_step('Iter')

                for key in loss_info:
                    if key == 'acc_mlm':
                        acc_mlm.update(loss_info[key])
                    if key == 'acc_match':
                        acc_match.update(loss_info[key])

                batch_time.update(time.time() - start)
                start = time.time()
                losses.update(loss.item())
                self.display(epoch, iteration, data_time, batch_time, losses, loss_info, acc_mlm, acc_match)
                iteration += 1
                
                if self.distributed:
                    dist.barrier()
                #val = self.eval(epoch)
                
            self.save_model(epoch)
            if not self.pre_train:
                val = self.eval(epoch)
                if(val is not None):
                    self.optim.scheduler_step('Epoch', val)
                self.scheduled_sampling(epoch)

            if self.distributed:
                dist.barrier()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument("--pre_train", action='store_true', help='default by False')
    parser.add_argument("--use_action", action='store_true', help='default by False')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.folder is not None:
        cfg_from_file(os.path.join(args.folder, 'config_server.yml'))
    cfg.ROOT_DIR = args.folder
    with open(os.path.join(args.folder,'eval_results.txt'),'a') as f:
        pass
    trainer = Trainer(args)
    trainer.train()
