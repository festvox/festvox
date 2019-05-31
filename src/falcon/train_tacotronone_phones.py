"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --data-root=<dir>         Directory contains preprocessed features.
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<dir>    Log Path [default: exp/log_tacotronOne]
    -h, --help                Show this help message and exit
"""
from docopt import docopt
from collections import defaultdict

# Use text & audio modules from existing Tacotron implementation.
import sys
from os.path import dirname, join
### This is not supposed to be hardcoded #####
FALCON_DIR = '/home/srallaba/projects/project_emphasis/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
##############################################

from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from utils.misc import *

from models import TacotronOne as Tacotron

import json

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, expanduser

import librosa.display
from matplotlib import pyplot as plt
import sys
import os
import tensorboard_logger
from tensorboard_logger import log_value
from hparams_arctic import hparams, hparams_debug_string

# Default DATA_ROOT
DATA_ROOT = join(expanduser("~"), "tacotron", "training")

fs = hparams.sample_rate

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

def train(model, data_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()
    #criterion_noavg = nn.L1Loss(reduction='none')

    global global_step, global_epoch
    while global_epoch < nepochs:
        running_loss = 0.
        for step, (x, input_lengths, mel, y) in tqdm(enumerate(data_loader)):
            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            x, mel, y = x[indices], mel[indices], y[indices]

            # Feed data
            x, mel, y = Variable(x), Variable(mel), Variable(y)
            if use_cuda:
                x, mel, y = x.cuda(), mel.cuda(), y.cuda()
            mel_outputs, linear_outputs, attn = model(
                x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    sorted_lengths, checkpoint_dir)
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm(
                model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("mel loss", float(mel_loss.item()), global_step)
            log_value("linear loss", float(linear_loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)

            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(data_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        #noavg_loss = criterion_noavg(mel_outputs, mel)
        #noavg_loss = noavg_loss.reshape(noavg_loss.shape[0], -1)
        #print("No avg output: ", torch.mean(noavg_loss, dim=-1))
        print("Loss: {}".format(running_loss / (len(data_loader))))


        global_epoch += 1


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint-path"]
    log_path = args["--log-event-path"]
    data_root = args["--data-root"]
    if data_root:
        DATA_ROOT = data_root

    # Override hyper parameters
    hparams.parse(args["--hparams"])

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Vocab size
    phids = make_phids(DATA_ROOT + '/txt.done.data.tacotron.phseq.train')
    outfile = checkpoint_dir + '/ids.json'
    with open(outfile, 'w') as outfile:
       json.dump(phids, outfile)
    print(phids)
    print("Length of vocabulary: ", len(phids))
    phids = dict(phids)
    #sys.exit()

    # Input dataset definitions
    X = FileSourceDataset(PhoneDataSource(DATA_ROOT, phids, "txt.done.data.tacotron.phseq.train"))
    Mel = FileSourceDataset(MelSpecDataSource(DATA_ROOT, "txt.done.data.tacotron.phseq.train"))
    Y = FileSourceDataset(LinearSpecDataSource(DATA_ROOT, "txt.done.data.tacotron.phseq.train"))

    # Dataset and Dataloader setup
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=1+ len(phids),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    model = model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"]
            global_epoch = checkpoint["global_epoch"]
        except:
            # TODO
            pass

    # Setup tensorboard logger
    tensorboard_logger.configure(log_path)

    print(hparams_debug_string())

    # Train!
    try:
        train(model, data_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)

