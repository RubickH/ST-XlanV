import pickle

video_len = pickle.load(open('gif_test_frame_I_dict.pkl','rb'))
f = open('test_I_length.txt','w')
minlen = 1000
maxlen = 0
video_len = dict(sorted(video_len.items(), key=lambda d: d[1]))
debug = 1 
for key,value in video_len.items():
    #key = './mm2020_test_videos/test_video_2020_'+str(i)+'.mp4'
    length = video_len[key]
    minlen = minlen if length>minlen else length
    maxlen = maxlen if length<maxlen else length
    f.write(key+' '+str(length)+'\n')
    debug = 1
debug = 1
f.write('minlen {} maxlen{}'.format(minlen, maxlen))
f.close()

