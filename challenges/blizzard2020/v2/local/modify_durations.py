import os, sys


'''
Takes as input: ph end_dur, start_dur
Generates output: ph dur_in_frames

'''

original_dir = sys.argv[1]
modified_dir = sys.argv[2]

frame_shift_ms = 12.5
sample_rate = 16000
frame_length_ms = 50

if not os.path.exists(modified_dir):
   os.makedirs(modified_dir)

files = sorted(os.listdir(original_dir))
for file in files:
 if file[0] =='b':
   fname = os.path.basename(file)
   g = open(modified_dir + '/' + fname, 'w')
   f = open(original_dir + '/' + file)
   for line in f:
     line = line.split('\n')[0].split()
     phone, end_dur, start_dur = line
     print(phone, end_dur, start_dur)
     end_dur = int(float(end_dur) * sample_rate / 200)
     start_dur =  int(float(start_dur) * sample_rate / 200)
     dur = end_dur - start_dur
     g.write(phone + ' ' + str(dur) + '\n')
   g.close()
   f.close()
