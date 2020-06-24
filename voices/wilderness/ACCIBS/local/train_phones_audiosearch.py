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
from model import TacotronOneSeqwiseAudiosearch as Tacotron

from torch import autograd
import random

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
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate




def validate(val_loader):

         with torch.no_grad():
             predictions = []
             targets = []
             for step, (x, pos, neg) in tqdm(enumerate(val_loader)):     

                # Feed data
                x, pos, neg = Variable(x), Variable(pos), Variable(neg)
                if use_cuda:
                    x, pos, neg = x.cuda(), pos.cuda(), neg.cuda()

                positive_labels = pos.new(pos.shape[0]).zero_() + 1
                negative_labels = pos.new(neg.shape[0]).zero_()

                logits_positive, x_reconstructed  = model(pos.long(), x)
                logits_negative, x_reconstructed  = model(neg.long(), x)

                classes = return_classes(logits_positive.view(-1, logits_positive.shape[-1]))
                classes = classes.cpu().numpy().tolist()
                predictions += classes
                targets += positive_labels.detach().cpu().numpy().tolist()

                classes = return_classes(logits_negative.view(-1, logits_negative.shape[-1]))
                classes = classes.cpu().numpy().tolist()
                predictions += classes
                targets += negative_labels.detach().cpu().numpy().tolist()

             get_metrics(predictions, targets)
                      

def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.CrossEntropyLoss()
    validate(val_loader)
    #sys.exit()

    global global_step, global_epoch
    while global_epoch < nepochs:
     with autograd.detect_anomaly():
        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (x, pos, neg) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Feed data
            x, pos, neg = Variable(x), Variable(pos), Variable(neg)
            if use_cuda:
                x, pos, neg = x.cuda(), pos.cuda(), neg.cuda()

            positive_labels = pos.new(pos.shape[0]).zero_() + 1
            negative_labels = pos.new(neg.shape[0]).zero_()

            # Multi GPU Configuration
            if use_multigpu:
               outputs,  r_, o_ = data_parallel_workaround(model, (x, mel))
               mel_outputs, linear_outputs, attn = outputs[0], outputs[1], outputs[2]
 
            else:
                inps = [pos, neg]
                labels = [positive_labels, negative_labels] 
                choice = random.choice([0, 1])
                choice_inputs = inps[choice]
                choice_labels = labels[choice]
                #print("Shape of choice: ", choice.shape)
                logits, x_reconstructed  = model(choice_inputs.long(), x)

            # Loss
            loss_search = criterion(logits.contiguous().view(-1, 2), choice_labels.long())
            #print("Shapes of x and x_recon: ", x.shape, x_reconstructed.shape) 
            loss_reconstruction = criterion(x_reconstructed.contiguous().view(-1, 1 + len(ph_ids)), x.long().contiguous().view(-1) )
            loss = loss_search + loss_reconstruction


            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("search loss", float(loss_search.item()), global_step)
            log_value("reconstruction loss", float(loss_reconstruction.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)
            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) + '\n')
        h.close()
        #sys.exit()

        global_epoch += 1

        validate(val_loader)


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

    ph_ids = dict(ph_ids)
    print(ph_ids)

    idsdict_file = checkpoint_dir + '/ids_phones.json'

    with open(idsdict_file, 'w') as outfile:
       json.dump(ph_ids, outfile)



    feats_name = 'phones'
    X_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name, ph_ids)
    X_val = categorical_datasource( vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name, ph_ids)

    # Dataset and Dataloader setup
    trainset = AudiosearchDataset(X_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_audiosearch, pin_memory=hparams.pin_memory)

    valset = AudiosearchDataset(X_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_audiosearch, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=1+ len(ph_ids),
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


