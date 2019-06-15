# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesize_tacotrononepy [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    -h, --help               Show help message.
"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from os.path import dirname, join
from utils import audio
from utils.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

from models import TacotronOne as Tacotron

from hparams_arctic import hparams

from tqdm import tqdm

import json


use_cuda = torch.cuda.is_available()


def tts(model, text):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()
    # TODO: Turning off dropout of decoder's prenet causes serious performance
    # regression, not sure why.
    # model.decoder.eval()
    model.encoder.eval()
    model.postnet.eval()

    sequence = np.array(text)
    #sequence = np.array(text_to_sequence(text, [hparams.cleaners]))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if use_cuda:
        sequence = sequence.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments = model(sequence)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio.denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram


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
    with open(checkpoints_dir + '/ids.json') as  f:
       charids = json.load(f)

    model = Tacotron(n_vocab=len(charids)+1,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)
    with open(checkpoints_dir + '/ids.json') as  f:
       charids = json.load(f)
    charids = dict(charids)

    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            fname = line.decode("utf-8").split()[0]
            text =  ' '.join(k.lower() for k in line.decode("utf-8").split()[1:])
            text = re.sub(r'[^\w\s]','', ''.join(k for k in text)) 
            words = nltk.word_tokenize(text)
            step = os.path.basename(checkpoint_path).split('.')[0].split('_')[-1]
            fname += '_' + step
            print(text, fname)
            text = '< ' + text + ' >' 
            textcheck = []  
            for c in text:
                if c in charids.keys():
                    textcheck.append(c)
                else:
                    textcheck.append('UNK')
            text = [charids[l] for l in textcheck]
            print("{}: {} ({} chars, {} words)".format(idx, text, len(text), len(words)))
            print(text, textcheck, fname)
            #sys.exit()

            waveform, alignment, _ = tts(model, text)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(fname))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="tacotron, {}".format(checkpoint_path))
            audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
