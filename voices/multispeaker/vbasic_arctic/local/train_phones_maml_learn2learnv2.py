"""Trainining script for Tacotron speech synthesis model. Trains MAML using learn2learn 
Uses same model

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
import copy
import learn2learn
from collections import defaultdict

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
##############################################
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from util import *
from model import TacotronOneSeqwise as Tacotron
from model import sgd_maml

import json

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from torch import autograd
#from torchsummary import summary

from os.path import join, expanduser

import tensorboard_logger
from tensorboard_logger import *
from hyperparameters import hparams, hparams_debug_string

vox_dir ='vox'

global_step = 0
global_epoch = 0
global_step_meta = 0
global_epoch_meta = 0

use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate



def finetune(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()

    global global_step, global_epoch
    while global_epoch < nepochs:
        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (x, spk, input_lengths, mel, y) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            x, spk, mel, y = x[indices], spk[indices], mel[indices], y[indices]

            # Feed data
            x, spk, mel, y = Variable(x), Variable(spk), Variable(mel), Variable(y)
            if use_cuda:
                x, spk, mel, y = x.cuda(), spk.cuda(), mel.cuda(), y.cuda()

            #mel_outputs, linear_outputs, attn = model(x, spk, mel, input_lengths=sorted_lengths)
            mel_outputs, linear_outputs, attn = model(x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss


            if global_step > 0 and global_step % hparams.save_states_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    None, checkpoint_dir)
                #visualize_phone_embeddings(model, checkpoint_dir, global_step)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("mel loss", float(mel_loss.item()), global_step)
            log_value("linear loss", float(linear_loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)
            log_histogram("Last Linear Weights", model.last_linear.weight.detach().cpu(), global_step)
            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) + '\n')
        h.close() 
        #sys.exit()

        global_epoch += 1


def phi_eval(loader, model):
    criterion = nn.L1Loss()
    linear_dim = model.linear_dim
    phi_loss = None
    for step, (x, spk, input_lengths, mel, y) in enumerate(loader):

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()
        
            x, spk, mel, y = x[indices], spk[indices], mel[indices], y[indices]

            # Feed data
            x, spk, mel, y = Variable(x), Variable(spk), Variable(mel), Variable(y)
            if use_cuda:
                x, spk, mel, y = x.cuda(), spk.cuda(), mel.cuda(), y.cuda()
             
            #mel_outputs, linear_outputs, attn = model(x, spk, mel, input_lengths=sorted_lengths)
            mel_outputs, linear_outputs, attn = model(x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss
            #print("Phi loss is ", loss) 
            if phi_loss is None:
               phi_loss = loss
            else:
               phi_loss += loss

            # You prolly should not return here
            #if step == 2:
            return phi_loss


def train(theta_model, theta_loader, phi_loader, optimizer_main,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    theta_model.train()
    if use_cuda:
        theta_model = theta_model.cuda()
    linear_dim = theta_model.linear_dim
    grad_norm = 0
    criterion = nn.L1Loss()

    global global_step_meta, global_epoch_meta
    while global_epoch_meta < nepochs:
        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (x, spk, input_lengths, mel, y) in tqdm(enumerate(theta_loader)):

            # Decay Learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer_main.param_groups:
                param_group['lr'] = current_lr

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()
            x, spk, mel, y = x[indices], spk[indices], mel[indices], y[indices]

            # Feed data
            x, spk, mel, y = Variable(x), Variable(spk), Variable(mel), Variable(y)
            if use_cuda:
                x, spk, mel, y = x.cuda(), spk.cuda(), mel.cuda(), y.cuda()


            ### Clone model. Now we have three models. theta_model, thetamodel_clone, phi_model
            thetamodel_clone = theta_model.clone()

            ### Get outputs from cloned model
            #mel_outputs, linear_outputs, attn = thetamodel_clone(x, spk, mel, input_lengths=sorted_lengths)
            mel_outputs, linear_outputs, attn = thetamodel_clone(x, mel, input_lengths=sorted_lengths)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            loss = mel_loss + linear_loss


            ### Update thetamodel_clone
            thetamodel_clone.adapt(loss)
            #sys.exit()

            if global_step_meta % 3 == 1:

               ### Get phi loss and gradients
               phi_loss = phi_eval(phi_loader, thetamodel_clone)
               phi_loss.backward()
               grad_norm = torch.nn.utils.clip_grad_norm_(
                    theta_model.parameters(), clip_thresh)

               ### Update theta_model with gradients of phi_loss
               optimizer_main.step()

            # Logs
            log_value("meta loss", float(loss.item()), global_step_meta)
            log_value("meta mel loss", float(mel_loss.item()), global_step_meta)
            log_value("meta linear loss", float(linear_loss.item()), global_step_meta)
            log_value("meta gradient norm", grad_norm, global_step_meta)
            log_value("meta learning rate", current_lr, global_step_meta)
            log_histogram("Last Linear Weights from meta ", theta_model.last_linear.weight.detach().cpu(), global_step_meta)
            global_step_meta += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(theta_loader))
        log_value("meta loss (per epoch)", averaged_loss, global_epoch_meta)
        h.write("Meta Loss after epoch " + str(global_epoch_meta) + ': '  + format(running_loss / (len(theta_loader))) + '\n')
        h.close()
        #sys.exit()

        global_epoch_meta += 1
    return theta_model


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

    # Vocab size
    with open(vox_dir + '/' + 'etc/ids_phones.json') as  f:
       ph_ids = json.load(f)

    with open(vox_dir + '/' + 'etc/ids_speakers.json') as  f:
       spk_ids = json.load(f)

    ph_ids = dict(ph_ids)
    spk_ids = dict(spk_ids)


    phidsdict_file = checkpoint_dir + '/ids_phones.json'
    with open(phidsdict_file, 'w') as outfile:
       json.dump(ph_ids, outfile)

    spkidsdict_file = checkpoint_dir + '/ids_speakers.json'
    with open(spkidsdict_file, 'w') as outfile:
       json.dump(spk_ids, outfile)


    # fnames_file, desc_file, feat_name, feats_dict=None, spk_dict=None
    feats_name = 'phones'
    theta_X_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train.awb', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids)
    phi_X_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train.rms', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids)


    feats_name = 'speaker'
    theta_spk_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train.awb', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids,
                                      spk_dict = spk_ids)
    phi_spk_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train.rms', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids,
                                      spk_dict = spk_ids)


    # fnames_file, desc_file, feat_name
    feats_name = 'lspec'
    theta_Y_train = float_datasource(fnames_file = vox_dir + '/' + 'fnames.train.awb', 
                               desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                               feat_name = feats_name)
    phi_Y_train = float_datasource(fnames_file = vox_dir + '/' + 'fnames.train.rms', 
                               desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                               feat_name = feats_name)

    feats_name = 'mspec'
    theta_Mel_train = float_datasource(fnames_file = vox_dir + '/' + 'fnames.train.awb', 
                               desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                               feat_name = feats_name)
    phi_Mel_train = float_datasource(fnames_file = vox_dir + '/' + 'fnames.train.rms', 
                               desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                               feat_name = feats_name)
    # Dataset and Dataloader setup
    thetaset = MultispeakerDataset(theta_X_train, theta_spk_train, theta_Mel_train, theta_Y_train)
    phiset = MultispeakerDataset(phi_X_train, phi_spk_train, phi_Mel_train, phi_Y_train)

    theta_loader = data_utils.DataLoader(
        thetaset, batch_size=int(hparams.batch_size),
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_spk, pin_memory=hparams.pin_memory)

    phi_loader = data_utils.DataLoader(
        phiset, batch_size=int(hparams.batch_size),
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_spk, pin_memory=hparams.pin_memory)


    # Model
    theta_model = learn2learn.algorithms.MAML(Tacotron(n_vocab=1+ len(ph_ids),
                     embedding_dim=256,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     ), lr=0.01, allow_unused=True, first_order=True)
    theta_model = theta_model.cuda()

    #model = DataParallelFix(model)

    optimizer_theta = optim.Adam(theta_model.parameters(),
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
        theta_model = train(theta_model, theta_loader, phi_loader, optimizer_theta,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=10,
              clip_thresh=hparams.clip_thresh)

        ### Derive phi_model from theta_model
        #phi_model.last_linear.weight.data = theta_model.last_linear.weight.data
        #phi_model.last_linear.bias.data = theta_model.last_linear.bias.data
        phi_model = copy.deepcopy(theta_model)
        optimizer_phi = optim.Adam(phi_model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

        finetune(phi_model, phi_loader, phi_loader, optimizer_phi,
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


