import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import pretrainedmodels
from pretrainedmodels import utils
import cv2
from PIL import Image

C, H, W = 3, 224, 224


def extract_frames(video, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)    
    os.makedirs(dst)
    try:
        im = Image.open(video)
        while True:
            current = im.tell()
            img = im.convert('RGB') 
            img_resized = img.resize((224,224))
            img_resized.save('{}/{}.jpg'.format(dst, str(current)))
            im.seek(current + 1)
    except:
        pass  


def extract_feats(params, model, load_image_fn):
    global C, H, W
    model.eval()

    dir_fc = params['output_dir']
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    print("save video feats to %s" % (dir_fc))
    
    dir_logits = params['output_dir_logits']
    if not os.path.isdir(dir_logits):
        os.mkdir(dir_logits)
    print("save video logits to %s" % (dir_logits))
    
    video_list = glob.glob(os.path.join(params['video_path'], '*'))
    succ_num = 0
    for video in tqdm(video_list):
        video_id = video.split("/")[-1].split(".")[0]
        dst = params['model'] + '_' + video_id
        extract_frames(video, dst)

        image_list = sorted(glob.glob(os.path.join(dst, '*.jpg')))

    
        max_num = np.min([20, len(image_list)])
        samples = np.round(np.linspace(
            0, len(image_list) - 1, max_num))
        image_list = [image_list[int(sample)] for sample in samples]
        #print(len(image_list))
        #print(video_id,step,len(image_list))
        if(len(image_list) < 10):
            shutil.rmtree(dst)
            continue
        images = torch.zeros((len(image_list), C, H, W))
        for iImg in range(len(image_list)):
            img = load_image_fn(image_list[iImg])
            images[iImg] = img
        with torch.no_grad():
            fc_feats = model.features(images.cuda())
            logits = model.logits(fc_feats).squeeze()
        img_feats = fc_feats.squeeze().cpu().numpy()
        # Save the res152 features
        outfile = os.path.join(dir_fc, video_id + '.npy')
        np.save(outfile, img_feats)
        # Save the res152 logits
        outfile_logits = os.path.join(dir_logits, video_id + '.npy')
        np.save(outfile_logits, logits.cpu().numpy())
        # cleanup
        shutil.rmtree(dst)
        succ_num += 1
        if(succ_num % 20 == 0):
            print('successfully processed {} gifs'.format(succ_num))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='1',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--output_dir", dest='output_dir', type=str,
                        default='/data/ACTION-Res152/', help='directory to store features')
    parser.add_argument("--output_dir_logits", dest='output_dir_logits', type=str,
                        default='/data/ACTION-Res152-logits/', help='directory to store features')
    parser.add_argument("--n_feats", dest='n_feats', type=int, default=20,
                        help='how many frames to sampler per video')

    parser.add_argument("--video_path", dest='video_path', type=str,
                        default='/data/ACTION-data', help='path to video dataset')
    parser.add_argument("--model", dest="model", type=str, default='resnet152',
                        help='the CNN model you want to use to extract_feats')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)

    C, H, W = 3, 224, 224
    model = pretrainedmodels.resnet152(pretrained='imagenet')
    load_image_fn = utils.LoadTransformImage(model)

    
    #model.last_linear = utils.Identity()
    #model = nn.DataParallel(model)
    
    model = model.cuda()
    extract_feats(params, model, load_image_fn)
