"""Trainining script for Tacotron speech synthesis model.

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
from model import *

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
from tensorboard_logger import *
from hyperparameters import hparams, hparams_debug_string

vox_dir ='vox'

global_step = 0
global_epoch = 0
global_step_finetuning = 0
global_epoch_finetuning = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate



def validate_model_full(model, val_loader):
     print("Validating the model")
     model.eval()
     y_true = []
     y_pred = []
     ff = open(exp_dir + '/eval' ,'a')

     with torch.no_grad():
      for step, (x, mel, fname) in enumerate(val_loader):
          #print("Shape of input during validation: ", x.shape, mel.shape)    
          x, mel = Variable(x).cuda(), Variable(mel).cuda()
          logits = model(mel)
          targets = x.cpu().view(-1).numpy()
          y_true += targets.tolist()
          predictions = return_classes(logits) 
          y_pred += predictions.tolist()  
          #print(predictions, targets)
     #print(y_pred, y_true)
     recall = get_metrics(y_pred, y_true)
     print("Unweighted Recall for the validation set:  ", recall)
     print('\n')
     return recall



def phi_train(phi_model, train_loader):


        criterion = nn.CrossEntropyLoss()
        phi_model.train()

        for step, (x, mel, fname) in enumerate(train_loader):

            # Feed data
            x, mel = Variable(x), Variable(mel)
            if use_cuda:
                x, mel = x.cuda(), mel.cuda()

            valence_outputs = phi_model(mel)

            # Loss
            loss = criterion(valence_outputs, x)
            #print("In meta training inner loop")
 
            # You prolly should not return here
            return loss



def finetune_train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    if use_cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    global global_step_finetuning, global_epoch_finetuning
    #validate_model(model, val_loader)
    while global_epoch_finetuning < nepochs:
        model.train()
        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (x, mel, fname) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step_finetuning)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Feed data
            x, mel = Variable(x), Variable(mel)
            if use_cuda:
                x, mel = x.cuda(), mel.cuda()

            val_outputs = model(mel)

            # Loss
            loss = criterion(val_outputs, x)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            if global_step_finetuning % checkpoint_interval == 0:

               save_checkpoint(
                    model, optimizer, global_step_finetuning, checkpoint_dir, global_epoch_finetuning)

            # Logs
            log_value("Finetune Training Loss", float(loss.item()), global_step_finetuning)
            log_value("gradient norm", grad_norm, global_step_finetuning)
            log_value("learning rate", current_lr, global_step_finetuning)
            global_step_finetuning += 1
            running_loss += loss.item()

            #print("In finetuning loop")

        averaged_loss = running_loss / (len(train_loader))
        log_value("fine tune loss (per epoch)", averaged_loss, global_epoch_finetuning)
        h.write("Finetune Loss after epoch " + str(global_epoch_finetuning) + ': '  + format(running_loss / (len(train_loader))) + '\n' + '\n')
        h.close()
        #log_value("Unweighted Recall per epoch", recall, global_epoch)
        #sys.exit()
        global_epoch_finetuning += 1

def meta_train(theta_model, phi_model, train_loader, val_loader, optimizer, optimizer_maml,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):

    running_loss_theta = 0.
    running_loss_phi = 0.
    criterion = nn.CrossEntropyLoss()
    global global_step, global_epoch
    #validate_model(model, val_loader)
    while global_epoch < nepochs:
        theta_model.train()
        phi_model.train()

        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (x, mel, fname) in tqdm(enumerate(train_loader)):

            model_copy = copy.deepcopy(theta_model)

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            optimizer_maml.zero_grad()

            # Feed data
            x, mel = Variable(x), Variable(mel)
            if use_cuda:
                x, mel = x.cuda(), mel.cuda()

            nce_loss = theta_model(mel)
            loss_thetamodel = nce_loss

            # Update
            nce_loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 theta_model.parameters(), clip_thresh)
            #fast_weights = optimizer_maml.step_maml()
            #for fwg in fast_weights:
            #    for p in fwg:
            #        print(p)
            #sys.exit()
            optimizer_maml.step()
            #for p in phi_model.encoder:
            #    print(p.shape)
            phi_model.encoder = copy.deepcopy(theta_model.encoder)
            theta_model = model_copy

            loss_phimodel = phi_train(phi_model, train_loader)
            loss_phimodel.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 theta_model.parameters(), clip_thresh)
            optimizer.step()

            if global_step % checkpoint_interval == 0:

               save_checkpoint(
                    theta_model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Logs
            log_value("Phi Loss", float(loss_phimodel.item()), global_step)
            log_value("Theta Loss", float(loss_thetamodel.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)
            global_step += 1
            running_loss_phi += loss_phimodel.item()
            running_loss_theta += loss_thetamodel.item()
            #print("In meta training outer loop")


        averaged_loss = running_loss / (len(train_loader))
        log_value("theta loss (per epoch)", averaged_loss, global_epoch)
        h.write("Theta Loss after epoch " + str(global_epoch) + ': '  + format(running_loss_theta / (len(train_loader))) 
                  + " Phi Loss: " + format(running_loss_phi / (len(train_loader)))
                  + '\n')
        h.close()
        #recall = validate_model_full(model, val_loader)
        #log_value("Unweighted Recall per epoch", recall, global_epoch)
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

    ff = open(exp_dir + '/eval' ,'w')
    ff.close()

    feats_name = 'lid'
    X_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    X_val = categorical_datasource(vox_dir + '/' +  'fnames.val', vox_dir + '/' +  'etc/falcon_feats.desc', feats_name,  vox_dir + '/' +  'festival/falcon_' + feats_name)

    feats_name = 'mfcc'
    Mel_train = float_datasource(vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Mel_val = float_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    # Dataset and Dataloader setup
    trainset = LIDDataset(X_train, Mel_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lid, pin_memory=hparams.pin_memory)

    valset = LIDDataset(X_val, Mel_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=4,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lid, pin_memory=hparams.pin_memory)

    # Model
    phi_model = LIDSeq2SeqDownsampling(39)
    phi_model = phi_model.cuda()

    theta_model = CPCBaseline(n_vocab=257,
                     embedding_dim=256,
                     mel_dim=39,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     )
    theta_model = theta_model.cuda()

    optimizer = optim.Adam(theta_model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)
    optimizer_maml = sgd_maml(params = theta_model.parameters(), lr=0.01)

    optimizer_finetune = optim.Adam(phi_model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    for name, module in theta_model.named_children():
        print(name)
    print('\n')
    for name, module in phi_model.named_children():
        print(name)


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
       print("Meta Training")
       meta_train(theta_model, phi_model, train_loader, val_loader, optimizer, optimizer_maml,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=1,
              clip_thresh=hparams.clip_thresh)
       print("Finetuning")
       finetune_train(phi_model, train_loader, val_loader, optimizer_finetune,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=1,
              clip_thresh=hparams.clip_thresh)

       recall = validate_model_full(phi_model, val_loader)
       print("Final Recall: ", recall)
    except Exception as e:
       print(e)
       sys.exit()

    print("Finished")
    sys.exit(0)


