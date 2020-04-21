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
from model import WaveLSTM5


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




def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    model.train()
    if use_cuda:
        model = model.cuda()
    #linear_dim = model.linear_dim

    criterion = DiscretizedMixturelogisticLoss() 

    # https://github.com/r9y9/wavenet_vocoder/blob/c4c148792c6263afbedb9f6bf11cd552668e26cb/train.py#L793
    if hparams.exponential_moving_average is not None:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None

    global global_step, global_epoch
    while global_epoch < nepochs:
        h = open(logfile_name, 'a')
        running_loss = 0.
        for step, (mel, x, spk) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Feed data
            x, spk, mel = Variable(x), Variable(spk), Variable(mel)
            if use_cuda:
                x, spk, mel = x.cuda(), spk.cuda(), mel.cuda()

            #print("Shapes of x, spk and mel: ", x.shape, spk.shape, mel.shape) 

            # Multi GPU Configuration
            if use_multigpu:
               outputs,  r_, o_ = data_parallel_workaround(model, (x, mel))
               mel_outputs, linear_outputs, attn = outputs[0], outputs[1], outputs[2]

            else:
                logits, targets = model(mel, spk, x)

            # Loss
            loss = criterion(logits.transpose(1,2), targets)


            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            if ema is not None:
              for name, param in model.named_parameters():
                 if name in ema.shadow:
                    ema.update(name, param.data)

            if global_step % checkpoint_interval == 0:

               save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch, ema=ema)
               #print("Saved ema")


            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)
            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) + '\n')
        h.close() 

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
    feats_name = 'r9y9inputmol'
    X_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids)

    feats_name = 'speaker'
    spk_train = categorical_datasource( fnames_file = vox_dir + '/' + 'fnames.train', 
                                      desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                                      feat_name = feats_name, 
                                      feats_dict = ph_ids,
                                      spk_dict = spk_ids)

    # fnames_file, desc_file, feat_name
    feats_name = 'r9y9outputmel'
    Mel_train = float_datasource(fnames_file = vox_dir + '/' + 'fnames.train', 
                               desc_file = vox_dir + '/' + 'etc/falcon_feats.desc', 
                               feat_name = feats_name)

    # Dataset and Dataloader setup
    trainset = MultispeakerVocoderDataset(X_train, spk_train, Mel_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_r9y9melNmolNspk, pin_memory=hparams.pin_memory)

    # Model
    model = WaveLSTM5()
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
        train(model, train_loader, train_loader, optimizer,
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


