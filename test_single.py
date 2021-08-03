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
from evaluation.evaler_gen_results import Evaler
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
            eval_annfile = cfg.INFERENCE.VAL_ANNFILE
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

    def setup_network(self):
        model = models.create(cfg.MODEL.TYPE, False, True)
        self.model = torch.nn.DataParallel(model).cuda()
        if self.args.resume >= 0:
            self.model.load_state_dict(
                torch.load(self.snapshot_path("caption_model", self.args.resume),
                    map_location=lambda storage, loc: storage), strict=False
            )
        debug = 1
        
    def eval(self, epoch):
        res, results = self.evaler(self.model, 'test_' + str(epoch))
        self.logger.info('######## Epoch ' + str(epoch) + ' ########')
        if(res is not None):
            self.logger.info(str(res))
        else:
            with open('upload_results/0622/caps_beam3.txt','w') as f:
                for i in range(len(results)):
                    imgid = str(results[i][cfg.INFERENCE.ID_KEY])
                    cap = results[i][cfg.INFERENCE.CAP_KEY]
                    f.write(imgid+' '+cap+'\n')
        self.logger.info('Evaluation Ends')   
                    

    def snapshot_path(self, name, epoch):
        snapshot_folder = os.path.join(cfg.ROOT_DIR, 'snapshot')
        return os.path.join(snapshot_folder, name + "_" + str(epoch) + ".pth")

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', default='/data/experiments/xlanv_msrvtt_scst', type=str)
    parser.add_argument("--resume", type=int, default=0)

    #if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if False:#args.folder is not None:
        cfg_from_file(os.path.join(args.folder,'config_server.yml'))
    else:
        cfg_from_file(os.path.join('config_test_msrvtt.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(args.resume)
