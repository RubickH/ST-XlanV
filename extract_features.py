import argparse
import os
import glob
import shutil
import subprocess
import numpy as np
import torch
from scipy import misc
from tqdm import tqdm

from matplotlib.pyplot import imread

import pdb
from model import S3D
import cv2
from PIL import Image


def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=256:384",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)

# def extract_frames(video, dst):
#     if os.path.exists(dst):
#         shutil.rmtree(dst)    
#     os.makedirs(dst)
#     try:
#         im = Image.open(video)
#         while True:
#             current = im.tell()
#             img = im.convert('RGB') 
#             img_resized = img.resize((256,384))
#             img_resized.save('{}/{}.jpg'.format(dst, str(current)))
#             im.seek(current + 1)
#     except:
#         pass  


def run(args):
    # Run RGB model
    file_weight = './S3D_kinetics400.pt'
    num_class = 400
    model = S3D(num_class)

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print ('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)

        print (' loaded')
    else:
        print ('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    # read the video list which records the readable video
    video_list = glob.glob(args.input_path + '*')
    succ_num = 0
    for vid in tqdm(video_list):
        try:
            extract_frames(vid, 'data/tmp_image')
            image_list = sorted(glob.glob(os.path.join('data/tmp_image', '*.jpg')))
            max_num = np.min([100, len(image_list)])
            samples = np.round(np.linspace( 0, len(image_list) - 1, max_num))
            image_list = [image_list[int(sample)] for sample in samples]

            #print(vid,len(image_list),step)

            
            if len(image_list) < 20:
                #print(vid,len(image_list),step)
                continue
            snippet = []
            for frame in image_list:
                img = cv2.imread(frame)
                img = img[...,::-1]
                snippet.append(img)
            clip = transform(snippet)
            #pdb.set_trace()
            with torch.no_grad():
                ps, features = model(clip.cuda())
            #pdb.set_trace()
            name = vid.split('_')[-1]
            name = name[:-4]
            name = name + '.npy'
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            if not os.path.exists(args.save_path_logits):
                os.makedirs(args.save_path_logits)
            np.save(args.save_path+name, features.cpu().numpy())
            np.save(args.save_path_logits+name, ps.cpu().numpy())
            succ_num += 1
            if(succ_num % 10 == 0):
                print(succ_num)
        except:
            print('error')
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RGB arguments
    parser.add_argument(
        '--rgb', action='store_true', help='Evaluate RGB pretrained network')
    parser.add_argument(
        '--rgb_weights_path',
        type=str,
        default='model_rgb.pth',
        help='Path to rgb model state_dict')

    parser.add_argument('--input_path', type=str, default='/home/v-yiqhuang/mycontainer/v-yiqhuang/mm2021_test_videos/')
    parser.add_argument('--save_path', type=str, default="/data/ACMMM-test/ACMMM-test-S3D/")
    parser.add_argument('--save_path_logits', type=str, default="/data/ACMMM-test/ACMMM-test-S3D-logits/")
    args = parser.parse_args()
    run(args)
