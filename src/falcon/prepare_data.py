import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
import re


'''Syntax
python3.5 prepare_data.py tdd_file dest_dir wav_dir
python3.5 prepare_data.py etc/txt.done.data data/all wav 
'''

tdd_file = 'etc/txt.done.data_nocarets'
dest_dir = '/home1/srallaba/challenges/blizzard2019/dataprep_tacotron_top2000_hpf/'
train_file = dest_dir + '/train.txt'
wav_dir = '/home1/srallaba/challenges/blizzard2019/voices/cmu_us_blzrd2019splitfileshpf_arctic/wav/'

tdd_file  = sys.argv[1]
dest_dir = sys.argv[2]
wav_dir = sys.argv[3]
if not os.path.exists(dest_dir):
   os.mkdir(dest_dir)
train_file = dest_dir + '/train.txt'
g = open(train_file , 'w')
g.close()
print(dest_dir)

_max_out_length = 700

f = open(tdd_file, encoding='utf-8')
for line in f:
 if len(line) > 2:
    line = line.split('\n')[0].split()
    line = ' '.join(k for k in line)
    line = re.sub(r'[^\w\s]','', line).strip().split()
    #print("Line is ", line)
    fname = line[0]
    text = ' '.join(k for k in line[1:-2])
    wav_fname = wav_dir + '/' + fname + '.wav'
    #print("Text is ", text)
    #print("Wave fname is : ", wav_fname)
    #sys.exit()

    wav = audio.load_wav(wav_fname)
    max_samples = _max_out_length * 5 / 1000 * 16000
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    lspec_fname = fname + '_lspec.npy'
    mspec_fname = fname + '_mspec.npy'
    print(fname, lspec_fname, mspec_fname)   

    np.save(dest_dir + '/' + lspec_fname, spectrogram.T, allow_pickle=False)
    np.save(dest_dir + '/' + mspec_fname, mel_spectrogram.T, allow_pickle=False)

    g = open(train_file, 'a')
    g.write(lspec_fname + '|' + mspec_fname + '|' + str(n_frames) + '| { ' + text + '}' + '\n')
    g.close()
