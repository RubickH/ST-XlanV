import os
import re
all_ids_res152 = set(os.listdir('../data/vlp_res152_feats/'))
all_ids_i3d = set(os.listdir('../data/vlp_i3d_feats/'))

all_ids = sorted(list(all_ids_res152.intersection(all_ids_i3d)))
all_ids.remove('89057.npy')
all_ids.remove('80919.npy')
all_ids.remove('119896.npy')


train_txt = open('../data/train_ids.txt','w')
val_txt = open('../data/val_ids.txt','w')

for i in range(len(all_ids)-2000):
    current_id = re.split('\.',all_ids[i])[0]
    train_txt.write(current_id+'\n')
    debug = 1

for i in range(len(all_ids)-2000,len(all_ids)):
    current_id = re.split('\.',all_ids[i])[0]
    val_txt.write(current_id+'\n')
    debug = 1
    
print(len(all_ids))
train_txt.close()
val_txt.close()