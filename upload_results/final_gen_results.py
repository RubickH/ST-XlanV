import json
import re
import numpy as np

ext = {'used': 'true', 'details': 'We now used both ACTION and MSRVTT to train the st-XLANV model 07.05'}
upload_file1 = {"version":"VERSION 1.0","result":[],"external_data":ext}
upload_file2 = {"version":"VERSION 1.0","result":[],"external_data":ext}
upload_file3 = {"version":"VERSION 1.0","result":[],"external_data":ext}

cap1 = open('0706/caps_models_last_1.txt').readlines()
cap2 = open('0706/caps_models_old_bm4.txt').readlines()
cap3 = open('0706/caps_models_old_bm5.txt').readlines()
cap4 = open('0706/caps_models_new_bm4.txt').readlines()
cap5 = open('0706/caps_models_new_bm5.txt').readlines()
cap6 = open('0706/caps_models_new2_bm5.txt').readlines()
cap7 = open('0706/caps_models_new2_bm4.txt').readlines()
cap8 = open('0706/caps_models_new3_bm5.txt').readlines()
cap9 = open('0706/caps_models_new4_bm4.txt').readlines()
cap10 = open('0706/caps_models_new4_bm5.txt').readlines()
cap11 = open('0706/caps_models_new5_bm4.txt').readlines()
cap12 = open('0706/caps_models_new5_bm5.txt').readlines()



for cnt in range(3000):
    video_id = 'test_video_2021_'+str(cnt)
    
    #all_caption = result[cnt]['caption']
    #split_caption = re.split("\.",all_caption)
    
    caption1 = re.split(' ',cap1[cnt].strip().lower())[1:]
    caption2 = re.split(' ',cap2[cnt].strip().lower())[1:]
    caption3 = re.split(' ',cap3[cnt].strip().lower())[1:]
    caption4 = re.split(' ',cap4[cnt].strip().lower())[1:]
    caption5 = re.split(' ',cap5[cnt].strip().lower())[1:]
    caption6 = re.split(' ',cap6[cnt].strip().lower())[1:]
    caption7 = re.split(' ',cap7[cnt].strip().lower())[1:]
    caption8 = re.split(' ',cap8[cnt].strip().lower())[1:]
    caption9 = re.split(' ',cap9[cnt].strip().lower())[1:]
    caption10 = re.split(' ',cap10[cnt].strip().lower())[1:]
    caption11 = re.split(' ',cap11[cnt].strip().lower())[1:]
    caption12 = re.split(' ',cap12[cnt].strip().lower())[1:]
  
    a = len(caption1)
    b = len(caption2)
    c = len(caption3)
    d = len(caption4)
    e = len(caption5)
    f = len(caption6)
    g = len(caption7)
    h = len(caption8)
    i = len(caption9)
    j = len(caption10)
    k = len(caption11)
    l = len(caption12)
  
    
    all_len = [-a,-b,-c,-d,-e,-f,-g,-h,-i,-j,-k,-l]
    all_cap = [caption1,caption2,caption3,caption4,caption5,caption6,caption7,caption8,caption9,caption10,caption11,caption12]
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
