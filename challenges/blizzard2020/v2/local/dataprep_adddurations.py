import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re


'''Syntax
python3.5 $FALCONDIR/dataprep_addmspec.py vox/fnames .

'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_durations'
phones_dir = vox_dir + '/festival/falcon_phones'
assure_path_exists(feats_dir)
assure_path_exists(phones_dir)

desc_file = vox_dir + '/etc/falcon_feats.desc'
durations_dir = vox_dir + '/durations_phones_modified'

f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:

    ctr += 1
    line = line.split('\n')[0]
    fname = line.split()[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split()[0]
    dur_fname = durations_dir + '/' + fname + '.dur'
    feats_fname = feats_dir + '/' + fname + '.feats'
    phones_fname = phones_dir + '/' + fname + '.feats'
    g = open(dur_fname)
    h = open(feats_fname, 'w')
    k = open(phones_fname, 'w')
    dur_seq = ''
    phseq ='<'
    c = 0
    for line in g:
        line = line.split('\n')[0]
        dur = line.split()[1]
        phone = line.split()[0]
        if c == 0:
           c = 1
           continue
        dur_seq += ' ' + dur
        phseq += ' ' + phone
    phseq += ' >'
    h.write(dur_seq + '\n')
    k.write(phseq + '\n')
    h.close() 
    g.close()
    k.close()



g = open(desc_file, 'a')
g.write('durations|single|categorical' + '\n')
g.close()


