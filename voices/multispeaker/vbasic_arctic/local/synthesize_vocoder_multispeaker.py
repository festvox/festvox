""" Synthesis waveform from trained model.

Usage: synthesize_tacotronone.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    --logits-dim=<N>         Dimensions for logits [default: 30] 
    -h, --help               Show help message.

"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
from os.path import dirname, join
os.environ['CUDA_VISIBLE_DEVICES']='1'
### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
festvox_dir = os.environ.get('FESTVOXDIR')

from utils import audio
from utils.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

from util import *
from model import WaveLSTM5

from hyperparameters import hparams

from tqdm import tqdm

import json

from scipy.io.wavfile import write

use_cuda = torch.cuda.is_available()


vox_dir ='vox'

spkcode = 0

def vocoder(model, mel, spk):

    if use_cuda:
        model = model.cuda()

    model.eval()

    sequence = np.array(mel)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    
    if use_cuda:
        sequence = sequence.cuda()
    spk = sequence.new(sequence.shape[0],1).zero_() + int(spk)

    waveform = model.forward_eval(sequence, spk)

    return waveform


if __name__ == "__main__":

    args = docopt(__doc__)
    print("Command line args:\n", args)

    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    logits_dim = int(args["--logits-dim"])

    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model = WaveLSTM5()
    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])

    os.makedirs(dst_dir, exist_ok=True)

    with open(vox_dir + '/' + 'etc/ids_speakers.json') as  f:
       spk_ids = json.load(f)


    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):


            fname = line.decode("utf-8").split()[0].zfill(8)
            spk_name = line.decode("utf-8").split()[1]

            spk = spk_ids[spk_name]

            cmd = 'cp ' + festvox_dir + '/voices/arctic/' + spk_name + '/vox/wav/' + fname + '.wav ' + dst_dir + '/' + spk_name + '_' + fname + '_original.wav'
            print(cmd)
            os.system(cmd)

            mel_fname = festvox_dir + '/voices/arctic/' + spk_name + '/vox/festival/falcon_r9y9outputmel/' + fname + '.feats.npy'
            mel = np.load(mel_fname) 

            wav_fname = festvox_dir + '/voices/arctic/' + spk_name + '/vox/festival/falcon_r9y9inputmol/' + fname + '.feats.npy'
            wav = np.load(wav_fname)
            assert len(wav) > 1
            waveform = wav
            print(np.max(wav), np.min(wav))
            dest_fname =  spk_name + '_' + fname + '_resynth'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(dest_fname, file_name_suffix))
            write(dst_wav_path, 16000, waveform)


            waveform = vocoder(model, mel, spk)
            dest_fname =  spk_name + '_' + fname + '_generated'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(dest_fname, file_name_suffix))
            write(dst_wav_path, 16000, waveform)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

