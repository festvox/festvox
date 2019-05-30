import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
import re


'''Syntax
python3.5 prepare_data.py tdd_file dest_dir wav_dir
python3.5 prepare_data.py etc/txt.done.data data/all wav 
python3.5 prepare_data.py etc/tdd .

tdd_file = 'etc/txt.done.data_nocarets'
dest_dir = '/home1/srallaba/challenges/blizzard2019/dataprep_tacotron_top2000_hpf/'
train_file = dest_dir + '/train.txt'
wav_dir = '/home1/srallaba/challenges/blizzard2019/voices/cmu_us_blzrd2019splitfileshpf_arctic/wav/'

'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
wav_dir = vox_dir + '/wav'
mspec_dir = vox_dir + '/mspec'
lspec_dir = vox_dir + '/lspec'
if not os.path.exists(mspec_dir): 
   os.mkdir(mspec_dir)
   os.mkdir(lspec_dir)
train_file = vox_dir + '/etc/txt.done.data.tacotron'
g = open(train_file , 'w')
g.close()

_max_out_length = 700

f = open(tdd_file, encoding='utf-8')
for line in f:
 if len(line) > 2:
    line = line.split('\n')[0].split()
    line = ' '.join(k for k in line)
    text = re.sub(r'[^\w\s]','', ' '.join(k for k in line.split()[1:])).strip().split()

    fname = line.split()[0]
    text = ' '.join(k.lower() for k in text)
    
    ### This is not a good fix - Sai Krishna 27 May 2019 #########
    # https://stackoverflow.com/questions/9942594/unicodeencodeerror-ascii-codec-cant-encode-character-u-xa0-in-position-20 
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = '< ' + text + ' >'  
    ##############################################################

    wav_fname = wav_dir + '/' + fname + '.wav'

    wav = audio.load_wav(wav_fname)
    max_samples = _max_out_length * 5 / 1000 * 16000
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    lspec_fname = lspec_dir + '/' + fname + '_lspec.npy'
    mspec_fname = mspec_dir + '/' + fname + '_mspec.npy'
    print(fname, lspec_fname, mspec_fname)   

    np.save(lspec_fname, spectrogram.T, allow_pickle=False)
    np.save(mspec_fname, mel_spectrogram.T, allow_pickle=False)

    g = open(train_file, 'a')
    g.write(lspec_fname + '|' + mspec_fname + '|' + str(n_frames) + '| ' + text  + '\n')
    g.close()
