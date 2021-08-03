import numpy as np
import os
import pdb
from tqdm import trange

ori_root = '/data/ACTION-Res152-filtered'
new_root_out = '/data/ACTION-Res152-final/'
file_list = os.listdir(ori_root)

set_num = 20
n_dim = 2048
for i in trange(len(file_list)):
    file = file_list[i]
    ori_array = np.mean(np.mean(np.load(os.path.join(ori_root,file)),axis=-1),axis=-1)
    cur_feat = ori_array.shape[0]

    feat_num = ori_array.shape[0]
    if(feat_num != set_num):
        repeat_num = int(np.ceil(set_num / feat_num))
        margin_num = set_num - (repeat_num-1)*feat_num
        new_array = np.zeros([set_num, n_dim])
        
        for i in range(feat_num):
            if(i < margin_num):
                for j in range(repeat_num):
                    cur_index = i*repeat_num + j 
                    new_array[cur_index] = ori_array[i]
            else:
                for j in range(repeat_num - 1):
                    cur_index = margin_num * repeat_num + (i-margin_num)*(repeat_num-1) + j
                    new_array[cur_index] = ori_array[i]

        np.save(new_root_out + file, new_array)
    else:            
        np.save(new_root_out + file, ori_array)
