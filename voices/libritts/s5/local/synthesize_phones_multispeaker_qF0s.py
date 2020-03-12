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
from model import TacotronOneSeqwiseMultispeakerqF0 as Tacotron

from hyperparameters import hparams

from tqdm import tqdm

import json


use_cuda = torch.cuda.is_available()


def get_textNqF0s(line, phids):
    line = line.decode("utf-8").split()[1:]
    texts = [phids['<']]
    qF0s = [0.0]
    for k in line:
        if '_' in k:
           phone = k.split('_')[0]
           texts.append(phids[phone])
           qF0 = float(k.split('_')[1])
           qF0s.append(qF0)
        else:
           phone = k
           texts.append(phids[phone])
           qF0s.append(4.0)

    texts += [phids['>']]
    qF0s += [0.0]
    assert len(texts) == len(qF0s) 
    return texts, qF0s


def tts(model, text, spk, qF0s):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()

    model.encoder.eval()
    model.postnet.eval()

    sequence = np.array(text)
    spk = np.array([spk])
    #sequence = np.array(text_to_sequence(text, [hparams.cleaners]))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    spk = Variable(torch.from_numpy(spk))
    qF0s = np.array(qF0s)
    qF0s =  Variable(torch.from_numpy(qF0s)).unsqueeze(0)

    if use_cuda:
        sequence = sequence.cuda()
        spk = spk.cuda()
        qF0s = qF0s.cuda()
 
    # Greedy decoding
    mel_outputs, linear_outputs, alignments = model(sequence, spk, qF0s.long())

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
    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phids = json.load(f)

    with open(checkpoints_dir + '/spk_ids') as  f:
       speakers_dict = json.load(f)


    model = Tacotron(n_vocab=len(phids)+1,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     num_spk=len(speakers_dict.keys())
                     )
    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)
    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phids = json.load(f)
    phids = dict(phids)

    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = max_decoder_steps


    ids2speakers = {v:k for (k,v) in speakers_dict.items()}
    speakers = list(speakers_dict.keys())
    print(ids2speakers)

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):

            fname = line.decode("utf-8").split()[0]
            fname = fname.split('_')
            fname_original = '_'.join(k for k in fname[1:])
            print(fname, fname_original)
            cmd = 'cp ' + 'vox/wav/' + '_'.join(k for k in fname) + '.wav ' + dst_dir + '/' + fname_original + '_original.wav'
            os.system(cmd)

            #text = ' '.join(k for k in line.decode("utf-8").split()[1:])
            #text = '< ' + text + ' >'
            #text = [phids[l] for l in text.split()]
            text, qF0s = get_textNqF0s(line, phids)

            # Generating from original speaker
            spk = speakers_dict[fname[0]]
            waveform, alignment, _ = tts(model, text, spk, qF0s)
            fname_generated = '_'.join(k for k in fname[1:])
            fname_generated = fname_generated + '_generated'
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname_generated, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(fname_generated))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="tacotron, {}".format(checkpoint_path))
            audio.save_wav(waveform, dst_wav_path)

            # Generating from a different speaker
            spk = np.random.randint(len(speakers))
            #fname = fname.split('_')
            #fname[0] = ids2speakers[spk]
            fname_transferred = '_'.join(k for k in fname[1:])
            fname_transferred = fname_transferred + '_transferred'
            print("I picked a random number as ", spk, " the corresponding speaker from the dictionary is ", ids2speakers[spk], " the filename I am storing is ", fname_transferred)
            print(text, fname_transferred)
            waveform, alignment, _ = tts(model, text, spk, qF0s)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname_transferred, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_alignment.png".format(fname_transferred))
            plot_alignment(alignment.T, dst_alignment_path,
                           info="tacotron, {}".format(checkpoint_path))
            audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

