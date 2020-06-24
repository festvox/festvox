import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re

 
'''Syntax
python3.5 $FALCONDIR/dataprep_addmspec.py etc/tdd .

'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
wavfeats_dir = vox_dir + '/festival/falcon_r9y9inputmol'
melfeats_dir = vox_dir + '/festival/falcon_r9y9outputmel'
wav_dir = vox_dir + '/wav'
assure_path_exists(wavfeats_dir)
assure_path_exists(melfeats_dir)
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

    wav, _ = librosa.effects.trim(wav, top_db=60, frame_length=2048, hop_length=512)
    if hparams.highpass_cutoff > 0.0:
        wav = audio.low_cut_filter(wav, hparams.sample_rate, hparams.highpass_cutoff)
 
    if hparams.global_gain_scale > 0:
        wav *= hparams.global_gain_scale
 
    # Clip
    wav = np.clip(wav, -1.0, 1.0)
 
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mspec_fname = melfeats_dir + '/' + fname + '.feats'
    np.save(mspec_fname, mel_spectrogram.T, allow_pickle=False)
    wav_fname = wavfeats_dir + '/' + fname + '.feats'
    np.save(wav_fname, wav)
 
 
 
 
g = open(desc_file, 'a')
g.write('r9y9outputmel|multi|float' + '\n')
g.write('r9y9inputmol|single|float' + '\n')
g.close()
 
 

