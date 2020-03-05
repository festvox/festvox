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
feats_dir = vox_dir + '/festival/falcon_quant'
wav_dir = vox_dir + '/wav'
assure_path_exists(feats_dir)

desc_file = vox_dir + '/etc/falcon_feats.desc'

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
    wav_fname = wav_dir + '/' + fname + '.wav'
    wav = audio.load_wav(wav_fname)
    quant = wav * (2**15 - 0.5) - 0.5
    quant_fname = feats_dir + '/' + fname + '.feats'
    np.save(quant_fname, quant, allow_pickle=False)



g = open(desc_file, 'a')
g.write('quants|single|int' + '\n')
g.close()


