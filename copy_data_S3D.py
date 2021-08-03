import numpy as np
import os
import pdb
from tqdm import trange

ori_root = '/data/ACTION-S3D-filtered'
new_root_out = '/data/ACTION-S3D-final/'
file_list = os.listdir(ori_root)

set_num = 12
n_dim = 1024
for i in trange(len(file_list)):
    file = file_list[i]
    ori_array = np.load(os.path.join(ori_root,file)).transpose(2,1,0,3,4).squeeze()
    feat_num = ori_array.shape[0]
    repeat_num = int(np.ceil(set_num / feat_num))
    margin_num = set_num - (repeat_num-1)*feat_num
    new_array = np.zeros([set_num, 1024])
    
    for i in range(feat_num):
        if(i < margin_num):
            for j in range(repeat_num):
                cur_index = i*repeat_num + j 
                new_array[cur_index] = ori_array[i]
                #print(i,cur_index)
        else:
            for j in range(repeat_num - 1):
                cur_index = margin_num * repeat_num + (i-margin_num)*(repeat_num-1) + j
                new_array[cur_index] = ori_array[i]
                #print(i,cur_index)
    np.save(new_root_out + file, new_array)            
    #if S3D_array.shape[2] > max_num:
    #    max_num = S3D_array.shape[2]
    #pdb.set_trace()
    
#print(max_num)