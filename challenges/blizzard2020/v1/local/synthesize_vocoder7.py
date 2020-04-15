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
#os.environ['CUDA_VISIBLE_DEVICES']='2'
### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from utils import audio
from utils.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np

#from util import *
from model import WaveLSTM7

from hyperparameters import hparams

from tqdm import tqdm

import json

from scipy.io.wavfile import write

use_cuda = torch.cuda.is_available()


vox_dir ='vox'

def vocoder(model, mel):

    if use_cuda:
        model = model.cuda()

    model.eval()

    sequence = np.array(mel)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if use_cuda:
        sequence = sequence.cuda()

    waveform = model.forward_eval(sequence)

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

    model = WaveLSTM7(n_vocab=257,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     logits_dim=logits_dim,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            fname = line.decode("utf-8").split()[0].zfill(8)
            cmd = 'cp vox/wav/' + fname + '.wav ' + dst_dir + '/' + fname + '_original.wav'
            print(cmd)
            os.system(cmd)

            mel_fname = vox_dir + '/festival/falcon_r9y9outputmel/' + fname + '.feats.npy'
            mel = np.load(mel_fname) 

            wav_fname = vox_dir + '/festival/falcon_r9y9inputmol/' + fname + '.feats.npy'
            wav = np.load(wav_fname)
            assert len(wav) > 1
            waveform = wav
            print(np.max(wav), np.min(wav))
            dest_fname =  fname + '_resynth'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(dest_fname, file_name_suffix))
            write(dst_wav_path, 16000, waveform)


            waveform = vocoder(model, mel)
            dest_fname =  fname + '_generated'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(dest_fname, file_name_suffix))
            write(dst_wav_path, 16000, waveform)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

