""" Synthesis waveform from trained model.

Usage: synthesize_tacotronone.py [options] <checkpoint_acousticmodel> <checkpoint_vocoder> <text_list_file> <dst_dir>

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
os.environ['CUDA_VISIBLE_DEVICES']='2'
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
from model import WaveLSTM
from model import TacotronOneSeqwise as Tacotron

from hyperparameters import hparams

from tqdm import tqdm

import json

from scipy.io.wavfile import write

use_cuda = torch.cuda.is_available()


vox_dir ='vox'

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

    mel_outputs, linear_outputs, alignments = model(sequence)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio.denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel_outputs = mel_outputs[0].cpu().data.numpy()


    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, mel_outputs


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

    checkpoint_path_vocoder = args["<checkpoint_vocoder>"]
    checkpoint_path_acousticmodel = args["<checkpoint_acousticmodel>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]

    checkpoint_acousticmodel = torch.load(checkpoint_path_acousticmodel)
    checkpoint_vocoder = torch.load(checkpoint_path_vocoder)

    checkpoints_dir = os.path.dirname(checkpoint_path_acousticmodel)

    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phids = json.load(f)
    phids = dict(phids)

    acousticmodel = Tacotron(n_vocab=len(phids)+1,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )

    vocoder_model = WaveLSTM(n_vocab=257,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )


    #checkpoint = torch.load(checkpoint_path)
    #checkpoints_dir = os.path.dirname(checkpoint_path)

    acousticmodel.load_state_dict(checkpoint_acousticmodel["state_dict"])
    acousticmodel.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    vocoder_checkpoint_path = 'exp/exp_vocoding_bsz4seqlen8_cloneofwavernn/checkpoints/checkpoint_step1400000.pth'
    vocoder_model.load_state_dict(checkpoint_vocoder["state_dict"])

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            fname = line.decode("utf-8").split()[0].zfill(8)
            cmd = 'cp vox/wav/' + fname + '.wav ' + dst_dir + '/' + fname + '_original.wav'
            print(cmd)
            os.system(cmd)
            text = ' '.join(k for k in line.decode("utf-8").split()[1:])
            text = '< ' + text + ' >'
            print(text, fname)
            text = [phids[l] for l in text.split()]
            waveform, alignment, mel = tts(acousticmodel, text)
            waveform_vocoder = vocoder(vocoder_model, mel)
            print(waveform_vocoder.shape)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(fname))
            plot_alignment(alignment.T, dst_alignment_path, info="tacotron, {}".format(checkpoint_path_acousticmodel))
            audio.save_wav(waveform, dst_wav_path)

            dest_fname =  fname + '_generated_vocoder'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(dest_fname, file_name_suffix))
            write(dst_wav_path, 16000, waveform_vocoder)


    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

