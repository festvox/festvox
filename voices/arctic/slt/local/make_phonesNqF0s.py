""" Make phones and quantized F0s

Usage: local/make_phonesNqF0s.py [options] <phone_durations_dir> <qF0s_dir> <qF0sNphones_dir>

options:
    --factor=<N>  factor to convert frames to samples [default: 200].
    -h, --help               Show help message.

"""
from docopt import docopt
import os, sys
import numpy as np

'''Description
-> Expects a directory durations_phones_modified to exist. Each file in this directory contains three columns: phone start_duration end_duration
-> Creates the directory qF0sNphones if does not exist
'''


args = docopt(__doc__)
phones_durations_dir = args['<phone_durations_dir>']
qf0sNphones_dir = args['<qF0sNphones_dir>']
if not os.path.exists(qf0sNphones_dir):
   os.mkdir(qf0sNphones_dir)
qf0s_dir = args['<qF0s_dir>']
factor = int(args['--factor'])

files = sorted(os.listdir(phones_durations_dir))
print(files)
for i, file in enumerate(files):
   fname = file.split('.')[0]
   qf0_file = qf0s_dir + '/' + fname + '.f0'
   qf0 = np.loadtxt(qf0_file)
   f = open(phones_durations_dir + '/' + file)
   g = open(qf0sNphones_dir + '/' + fname + '.qF0sNphones', 'w')
   for line in f:
     line = line.split('\n')[0].split()
     phone = line[0]
     start_dur = int(float(line[1]) * factor)
     end_dur = int(float(line[2]) * factor)
     if start_dur == end_dur:
        continue
     qf0_segment = qf0[start_dur:end_dur]
     #print(phone, qf0_segment, start_dur, end_dur)
     g.write(phone + ' ' + str(max(qf0_segment)) + '\n')
   #print(word, start_dur, end_dur, max(qf0_segment))
   g.close()
   f.close()

   if i% 100 == 1:
      print("Processed ", i, " files out of ", len(files))

