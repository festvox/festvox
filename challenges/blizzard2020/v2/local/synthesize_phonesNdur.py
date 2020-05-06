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
from model import DurationAcousticModel as Tacotron

from hyperparameters import hparams

from tqdm import tqdm

import json


use_cuda = torch.cuda.is_available()
vox_dir = 'vox'

def tts(model, text):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()

    model.encoder.eval()
    model.postnet.eval()

    sequence = np.array(text)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if use_cuda:
        sequence = sequence.cuda()

    mel_outputs, linear_outputs = model(sequence)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio.denormalize(linear_output)

    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, spectrogram


if __name__ == "__main__":

    args = docopt(__doc__)
    print("Command line args:\n", args)

    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]

    #checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)
    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phids = json.load(f)

    model = Tacotron(n_vocab=len(phids)+1)
    checkpoint = torch.load(checkpoint_path)
    #checkpoints_dir = os.path.dirname(checkpoint_path)
    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phids = json.load(f)
    phids = dict(phids)

    model.load_state_dict(checkpoint["state_dict"])
    #model.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):

            fname = line.decode("utf-8").split()[0].zfill(8)
            cmd = 'cp vox/wav/' + fname + '.wav ' + dst_dir + '/' + fname + '_original.wav'
            print(cmd)
            os.system(cmd)

            # Load phones
            phones_fname = vox_dir + '/festival/falcon_phones/' + fname + '.feats'
            h = open(phones_fname)
            for l in h:
                l = l.split('\n')[0].split()
            phones = np.array([phids[k] for k in l])

            # Load durations
            durations_fname = vox_dir + '/festival/falcon_durations/' + fname + '.feats'
            h = open(durations_fname)
            for l in h:
                l = l.split('\n')[0].split()
            durations = l

            #print("Length of phones and durations: ", len(phones), len(durations.split()))
            assert len(phones) == len(durations) + 2

            # Extend durations
            durations_extended = []
            for i, d in enumerate(durations):
               d = int(d)
               while d > 0:
                 durations_extended += [i+1]
                 d = d -1
            phones = phones[durations_extended]

            #sys.exit()

            waveform, spectrogram = tts(model, phones)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname + '_tacotron', file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(fname))
            audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

