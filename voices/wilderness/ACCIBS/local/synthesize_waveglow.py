""" Synthesis waveform from trained model.

Usage: synthesize_tacotronone.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    -h, --help               Show help message.

"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
from os.path import dirname, join

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from utils import audio
from utils.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

#from util import *
from model import WaveGlowNVIDIA

from hyperparameters import hparams

from tqdm import tqdm

import json
from scipy.io.wavfile import write


use_cuda = torch.cuda.is_available()
vox_dir = 'vox'
MAX_WAV_VALUE = 32768.0


def vocoder(model, mel):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()

    model.eval()

    sequence = np.array(mel)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if use_cuda:
        sequence = sequence.cuda()

    with torch.no_grad():
       audio = model.infer(sequence)
    audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()

    waveform = audio.cpu().numpy()
    waveform = waveform.astype('int16')

    return waveform


if __name__ == "__main__":

    args = docopt(__doc__)
    print("Command line args:\n", args)

    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]

    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model = WaveGlowNVIDIA()
    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "r") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            fname = line.split()[0]
            print(fname)

            mel_fname = vox_dir + '/festival/falcon_r9y9outputmel/' + fname + '.feats.npy'
            mel = np.load(mel_fname)
            waveform = vocoder(model, mel)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname + '_waveglow', file_name_suffix))
            write(dst_wav_path, 16000, waveform)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

