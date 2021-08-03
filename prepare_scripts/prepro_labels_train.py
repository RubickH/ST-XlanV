import pickle
import json
import numpy as np
import re
#x = pickle.load(open('/media/rubick/part2/X-transformer/mscoco/sent/coco_train_input.pkl','rb'))
#y = pickle.load(open('/media/rubick/part2/X-transformer/mscoco/sent/coco_train_target.pkl','rb'))
max_length = 20

train_ids = open('../data_msrvtt_standard/train_ids.txt').readlines()
all_captions = json.load(open('../data_common/caption_msrvtt.json'))

encoded_num = 0
fail_num = 0
input_dict = {}
for i in range(len(train_ids)):
    train_input = []
    current_id = train_ids[i].strip()
    cat_id = 'video'+current_id
    caps = all_captions[cat_id]['captions']
    for j in range(20):
        #sent = []
        #cap = re.split(' ',caps[j])
        #for word in cap:
            #sent.append(word)
        train_input.append(caps[j])
    input_dict[current_id] = train_input
    encoded_num += 1
        
pickle.dump(input_dict,open('../data_msrvtt_standard/msrvtt_train_input.pkl','wb'))

print(encoded_num)
print(fail_num)
debug = 1
