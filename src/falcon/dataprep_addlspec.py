import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re


'''Syntax
python3.5 $FALCONDIR/dataprep_addtext.py etc/tdd .

'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_lspec'
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
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    lspec_fname = feats_dir + '/' + fname + '.feats'
    np.save(lspec_fname, spectrogram.T, allow_pickle=False)



g = open(desc_file, 'a')
g.write('lspec|multi|float' + '\n')
g.close()

