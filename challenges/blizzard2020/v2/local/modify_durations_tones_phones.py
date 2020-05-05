import os, sys


'''
Takes as input: punc, word, dur #0 zai 0.30500001 
Generates output: phone+tone dur_in_frames # zai_4 24
'''

original_dir = sys.argv[1]
modified_dir = sys.argv[2]

tones_file = '../../../../../voices/v2/tdd.phonesNtones'
tones_file = 'tdd.underscores'
f = open(tones_file)
fnames2tones = {}
for line in f:
   line = line.split('\n')[0].split()
   fname = 'blizzard_' + line[0]
   content = ' '.join(k for k in line[1:])
   #content = content.replace(' pau ', ' ' )
   fnames2tones[fname] = content
#print(fnames2tones)

frame_shift_ms = 12.5
sample_rate = 16000
frame_length_ms = 50

if not os.path.exists(modified_dir):
   os.makedirs(modified_dir)

files = sorted(os.listdir(original_dir))
for file in files:
 if file[0] =='b':
   fname = os.path.basename(file)
   content = fnames2tones[fname.split('.dur')[0]].split()
   content_idx = 0
   g = open(modified_dir + '/' + fname, 'w')
   f = open(original_dir + '/' + file)
   for line in f:
     line = line.split('\n')[0].split()
     print("I read : ", line)
     punc, word, dur = line
     if punc != '0':
        phone = word + punc
     else:
        phone = word
     if phone != '0' and phone != ',':
        content_phone, content_tone = content[content_idx].split('_')
        while content_phone == 'pau':
           content_idx += 1
           content_phone, content_tone = content[content_idx].split('_')
        if content_phone == phone:
           tone = content_tone
           content_idx += 1
        else:
           print("Check this edge case. content phone and current phone: ", content_phone, phone)
           print(fname)
           print("Content: ", content)
           sys.exit()
     elif phone == '0':
        tone = 0
        phone = 'pau'
     elif phone == ',':
        tone = 0
        phone = 'pau'   
     else:
        tone = 0
        phone = 'pau'  
     #print(phone, end_dur, start_dur, tone)
     dur = int(float(dur) * sample_rate / 200)
     g.write(phone + '_' + tone + ' ' + str(dur) + '\n')
   g.close()
   f.close()
