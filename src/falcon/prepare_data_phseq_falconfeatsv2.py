import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re


'''Syntax
python3.5 $FALCONDIR/prepare_data_phseq_falconfeatsv2.py ehmm/etc/txt.phseq.data . 

'''

### Flags
generate_feats_flag = 0

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
wav_dir = vox_dir + '/wav'
mspec_dir = vox_dir + '/mspec'
lspec_dir = vox_dir + '/lspec'
feats_dir = vox_dir + '/festival/falcon_feats'
phones_dir = vox_dir + '/festival/falcon_phones'

assure_path_exists(mspec_dir)
assure_path_exists(lspec_dir)
assure_path_exists(feats_dir)
assure_path_exists(phones_dir)

if generate_feats_flag:
  data_file = vox_dir + '/etc/txt.done.data.tacotron.phseq'
  g = open(data_file , 'w')
  g.close()




desc_file = vox_dir + '/etc/falcon_feats.desc'
g = open(desc_file , 'w')
g.close()

_max_out_length = 700

f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    line = line.split('\n')[0]

    fname = line.split()[0]
    phones = ' '.join(k for k in line.split()[1:])

    if generate_feats_flag:
       wav_fname = wav_dir + '/' + fname + '.wav'
       wav = audio.load_wav(wav_fname)
       max_samples = _max_out_length * 5 / 1000 * 16000
       spectrogram = audio.spectrogram(wav).astype(np.float32)
       n_frames = spectrogram.shape[1]
       mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
       lspec_fname = lspec_dir + '/' + fname + '_lspec.npy'
       mspec_fname = mspec_dir + '/' + fname + '_mspec.npy'
       np.save(lspec_fname, spectrogram.T, allow_pickle=False)
       np.save(mspec_fname, mel_spectrogram.T, allow_pickle=False)

       g = open(data_file, 'a')
       g.write(lspec_fname + '|' + mspec_fname + '|' + str(n_frames) + '| ' + phones  + '\n')
       g.close()

       g = open(feats_dir + '/' + fname + '.feats', 'w')
       for phone in phones.split():
          g.write(phone + '\n')
       g.close()

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    g = open(phones_dir + '/' + fname + '.feats', 'w')
    ph = ' '.join(k for k in phones.split())
    g.write(ph + '\n')
    g.close()



g = open(desc_file, 'w')
g.write('phones|single|categorical' + '\n')
g.close()

