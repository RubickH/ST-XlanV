import json
import re
import numpy as np

ext = {'used': 'false', 'details': 'We now used MSRVTT to train the st-XLANV model 06.04'}
upload_file1 = {"version":"VERSION 1.0","result":[],"external_data":ext}
upload_file2 = {"version":"VERSION 1.0","result":[],"external_data":ext}
upload_file3 = {"version":"VERSION 1.0","result":[],"external_data":ext}

cap1 = open('0704/caps_models_general_modified.txt').readlines()
cap2 = open('0704/caps_models_cider_modified.txt').readlines()
cap3 = open('0704/caps_models_bleu_modified.txt').readlines()


for i in range(3000):
    video_id = 'test_video_2021_'+str(i)
    
    #all_caption = result[i]['caption']
    #split_caption = re.split("\.",all_caption)
    
    caption1 = re.split(' ',cap1[i].strip().lower())[1:]
    caption2 = re.split(' ',cap2[i].strip().lower())[1:]
    caption3 = re.split(' ',cap3[i].strip().lower())[1:]
    
    a = len(caption1)
    b = len(caption2)
    c = len(caption3)
    
    all_len = [-a,-b,-c]
    all_cap = [caption1,caption2,caption3]
    sorted_len = np.argsort(all_len)
    debug = 1
    
    
    upload_file1['result'].append({"video_id":video_id,"caption":' '.join(all_cap[sorted_len[0]])})
    upload_file2['result'].append({"video_id":video_id,"caption":' '.join(all_cap[sorted_len[1]])})
    upload_file3['result'].append({"video_id":video_id,"caption":' '.join(all_cap[sorted_len[2]])})
    
    debug = 1
    
json.dump(upload_file1,open('result1.json','w'))
json.dump(upload_file2,open('result2.json','w'))
json.dump(upload_file3,open('result3.json','w'))
debug = 1
