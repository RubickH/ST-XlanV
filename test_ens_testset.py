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
from evaluation.evaler_ENS import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")
        
        self.setup_logging()
        self.setup_network()
        self.evaler = Evaler(
            eval_ids = cfg.DATA_LOADER.VAL_ID,
            s3d_feats = cfg.DATA_LOADER.VAL_S3D_FEATS,
            s3d_logits = cfg.DATA_LOADER.VAL_S3D_LOGITS,
            res152_feats = cfg.DATA_LOADER.VAL_RES152_FEATS,
            res152_logits = cfg.DATA_LOADER.VAL_RES152_LOGITS,            
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE,
            test = args.test
        )
           

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
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

    def parse_bool(self,str):
        if(str == 'true'):
            return True
        else:
            return False
    
    def setup_network(self):
        model_all = []
        
        selected_index = [0,2,4,5,6,8,9,10,11]   #selected_index = [0,1,2,3,4,5,6,7] best for old ens_models_final; [2,8,9,10] for 0706 models
        model_list = [599,595,595,589,587,582,580,578,606,606,600,599]
        pos_list = ['false','true','false','true','false','true','false','true','true','false','true','true']
        tag_list = ['true','true','false','false','false','false','false','false','true','false','true','true']
        

        
        
        for i in selected_index:
            model = models.create('XLANV', self.parse_bool(pos_list[i]),self.parse_bool(tag_list[i]))
            self.model = torch.nn.DataParallel(model).cuda()
            self.model.load_state_dict(
                    torch.load(self.snapshot_path("caption_model", model_list[i], pos_list[i], tag_list[i]),
                        map_location=lambda storage, loc: storage), strict=False
                )
            model_all.append(model)
        self.model = models.create('XLANV_ENS',model_all,None).cuda()
            

    def make_kwargs(self, indices, input_ids_all, input_seq, target_seq, s3d_feats, s3d_mask, res152_feats, res152_mask, tag_ids, tag_mask, aligned):
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
            cfg.PARAM.ALIGNED: aligned
        }
        return kwargs
    
    
    def eval(self):
        res, results = self.evaler(self.model, 'test_ENS' )
        self.logger.info('######## ENS models ########')
        if(res is not None):
            self.logger.info(str(res))
        else:
            with open('upload_results/0706/caps_models_new5_bm5.txt','w') as f:
                for i in range(len(results)):
                    imgid = str(results[i][cfg.INFERENCE.ID_KEY])
                    cap = results[i][cfg.INFERENCE.CAP_KEY]
                    f.write(imgid+' '+cap+'\n')
        self.logger.info('Evaluation Ends')   
                    

    def snapshot_path(self, name, epoch, pos, tag):
        snapshot_folder = '../ens_models_final'
        #snapshot_folder = '../0706-ens'
        if(tag == 'false'):
            return os.path.join(snapshot_folder, name + '_' + str(epoch) + '_' + pos + '.pth')
        else:
            return os.path.join(snapshot_folder, name + '_' + str(epoch) + '_' + pos + '_' + tag +'.pth')
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument("--test", type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    if(args.test == 0):
        cfg_from_file(os.path.join('config_test_msrvtt.yml'))
    else:
        cfg_from_file(os.path.join('config_test.yml'))

    tester = Tester(args)
    tester.eval()
