"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --conf=<json>             Path of configuration file (json).
    --exp-dir=<dir>           Experiment directory
    --gpu-id=<N>               ID of the GPU to use [default: 0]
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
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate


def validate_model(model, val_loader, epoch):
     print("Validating the model")
     model.eval()
     y_true = []
     y_pred = []
     fnames = []
     running_loss = 0.
     criterion = nn.CrossEntropyLoss()
     with torch.no_grad():
       for step, (feats, lid, lengths, fname) in enumerate(val_loader):

          sorted_lengths, indices = torch.sort(
               lengths.view(-1), dim=0, descending=True)
          sorted_lengths = sorted_lengths.long().numpy()
          feats, lid = feats[indices], lid[indices]
          #print(indices, fname)
          #fname = fname[indices.numpy()]
          indices = indices.numpy().tolist()
          fname = [f for index, f in sorted(zip(indices, fname))]

          feats, lid = Variable(feats), Variable(lid)
          feats, lid = feats.cuda(), lid.cuda().long()
          logits = model(feats)
          loss = criterion(logits, lid.long())
          running_loss += loss.item()
          targets = lid.cpu().view(-1).numpy()
          y_true += targets.tolist()
          predictions = return_classes(logits)
          y_pred += predictions.tolist()
          fnames += fname
          #print(fname)
     ff = open(exp_dir + '/eval_' + str(epoch).zfill(3) ,'a')
     assert len(fnames) == len(y_pred)
     for (f, yp, yt) in list(zip(fnames, y_pred, y_true)):
          if yp == yt:
            continue
          ff.write( f + ' ' + str(yp) + ' ' + str(yt) + '\n')
     ff.close()

     averaged_loss = running_loss / (len(val_loader))
     recall = get_metrics(y_pred, y_true)
     print("Validation Loss: ", averaged_loss)
     print("Unweighted Recall for the validation set:  ", recall)
     print('\n')
     return recall, model

if __name__ == "__main__":
    exp_dir = args["--exp-dir"]
    checkpoint_path = args["--checkpoint-path"]
    conf = args["--conf"]
    hparams.parse(args["--hparams"])

    # Override hyper parameters
    if conf is not None:
        with open(conf) as f:
            hparams.parse_json(f.read())


    lid_train, lid_val, _ = get_cat_feats('lid')
    fnames_train, fnames_val, _ = get_cat_feats('fnames')
    mfcc_train, mfcc_val = get_float_feats('mfcc')

    valset = LIDmfccsDataset(lid_val, mfcc_val, fnames_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=False,
        collate_fn=collate_fn_lidmfcc, pin_memory=hparams.pin_memory)

    # Model
    model = LIDMixtureofExpertsmfccattention()
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

    print(hparams_debug_string())

    # eval
    try:
        recall, model = validate_model(model, val_loader, 'eval001')
        print("Final Recall: ", recall)
        recall, model = validate_model(model, val_loader, 'eval002')
        print("Final Recall: ", recall)
        recall, model = validate_model(model, val_loader, 'eval003')
        print("Final Recall: ", recall)
 

    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)


