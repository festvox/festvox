"""Trainining script for Mandarin speech synthesis model.

usage: train.py [options]

options:
    --conf=<json>             Path of configuration file (json).
    --gpu-id=<N>               ID of the GPU to use [default: 0]
    --exp-dir=<dir>           Experiment directory
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<dir>    Log Path [default: exp/log_tacotronOne]
    -h, --help                Show this help message and exit
"""
import os, sys
from docopt import docopt
args = docopt(__doc__)
print("Command line args:\n", args)
gpu_id = args['--gpu-id']
print("Using GPU ", gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id


from collections import defaultdict

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
##############################################
from utils.misc import * # ha sab kuch import karlo
from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from util import * # modify
from model import DNSLSTM

import json

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from os.path import join, expanduser

import tensorboard_logger
from tensorboard_logger import *  # import X and then from X import *. naice!!
from hyperparameters import hparams, hparams_debug_string

from scipy.io.wavfile import write

# Seriously??? vox_dir='vox'? kuch bhi
vox_dir ='vox'

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate




def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.CrossEntropyLoss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        h = open(logfile_name, 'a')
        running_loss = 0.
        running_loss_coarse = 0.
        running_loss_fine = 0.
        for step, batch in tqdm(enumerate(train_loader)):

            for c in batch:
                c = Variable(c).cuda()
 
            # Get elements from batch
            mels_noisy,  coarse_clean, fine_clean, coarse_float_clean, fine_float_clean = batch

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            mels_noisy,  coarse_clean, fine_clean, coarse_float_clean, fine_float_clean = Variable(mels_noisy).cuda(), Variable(coarse_clean).cuda(), Variable(fine_clean).cuda(), Variable(coarse_float_clean).cuda(), Variable(fine_float_clean).cuda()

            coarse_logits, coarse_targets, fine_logits, fine_targets = model(mels_noisy, coarse_clean, coarse_float_clean, fine_clean, fine_float_clean)

            # Loss
            #print("Shape of logits and targets: ", coarse_outputs.shape, coarse.shape)
            coarse_loss = criterion(coarse_logits.contiguous().view(-1, 256), coarse_targets.contiguous().view(-1))
            fine_loss = criterion(fine_logits.contiguous().view(-1, 256), fine_targets.contiguous().view(-1))

            loss = coarse_loss + fine_loss
            #print(loss)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            if global_step % checkpoint_interval == 0:

               save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)



            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("coarse loss", float(coarse_loss.item()), global_step)
            global_step += 1
            running_loss += loss.item()
            running_loss_coarse += coarse_loss.item()
            running_loss_fine += fine_loss.item()

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Coarse Loss after epoch " + str(global_epoch) + ': '  + format(running_loss_coarse / (len(train_loader))) +   " Fine Loss after epoch " + str(global_epoch) + ': '  + format(running_loss_fine / (len(train_loader))) +  '\n')
        h.close() 
        #sys.exit()


        global_epoch += 1


if __name__ == "__main__":

    exp_dir = args["--exp-dir"]
    checkpoint_dir = args["--exp-dir"] + '/checkpoints'
    checkpoint_path = args["--checkpoint-path"]
    log_path = args["--exp-dir"] + '/tracking'
    conf = args["--conf"]
    hparams.parse(args["--hparams"])

    # Override hyper parameters
    if conf is not None:
        with open(conf) as f:
            hparams.parse_json(f.read())

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logfile_name = log_path + '/logfile'
    h = open(logfile_name, 'w')
    h.close()


    feats_name = 'quant'
    X_train = categorical_datasource( vox_dir + '/' + 'fnames.noisy.head.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    X_val = CategoricalDataSource(vox_dir + '/' +  'fnames.noisy.head.val', vox_dir + '/' +  'etc/falcon_feats.desc', feats_name,  feats_name)

    feats_name = 'mspec'
    Mel_train = float_datasource(vox_dir + '/' + 'fnames.noisy.head.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Mel_val = FloatDataSource(vox_dir + '/' + 'fnames.noisy.head.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    # Dataset and Dataloader setup
    trainset = DNSDataset(X_train, Mel_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=0, shuffle=True,
        collate_fn=collate_fn_mspecNquant, pin_memory=hparams.pin_memory)
    
    ## Ok champion, tell me where you are using this  
    valset = DNSDataset(X_val, Mel_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Model
    model = DNSLSTM(n_vocab=257,
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    model = model.cuda()
    #model = DataParallelFix(model)

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
            global_step = int(checkpoint["global_step"])
            global_epoch = int(checkpoint["global_epoch"])
        except:
            print("Houston! We have got problems")
            sys.exit()
            

    # Setup tensorboard logger
    tensorboard_logger.configure(log_path)

    print(hparams_debug_string())

    # Train!
    try:
        train(model, train_loader, val_loader, optimizer,
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


