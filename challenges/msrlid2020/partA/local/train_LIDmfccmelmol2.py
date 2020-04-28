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


def validate_model(model, val_loader):
     print("Validating the model")
     model.eval()
     y_true = []
     y_pred = []
     fnames = []
     running_loss = 0.
     criterion = nn.CrossEntropyLoss()
     with torch.no_grad():
       for step, (mfcc, mfcc_lengths, mel, mol, lid, fname) in enumerate(val_loader):

          # Sort by length
          sorted_lengths, indices = torch.sort(
             mfcc_lengths.view(-1), dim=0, descending=True)
          sorted_lengths = sorted_lengths.long().numpy()
  
          mfcc, mel, mol, lid = mfcc[indices], mel[indices], mol[indices], lid[indices]
          mfcc, mel, mol, lid = Variable(mfcc).cuda(), Variable(mel).cuda(), Variable(mol).cuda(), Variable(lid).cuda()
          logits, vq_penalty, encoder_penalty, entropy = model(mfcc, mel, mol)
          loss = criterion(logits, lid.long())
          running_loss += loss.item()
          targets = lid.cpu().view(-1).numpy()
          y_true += targets.tolist()
          predictions = return_classes(logits)
          y_pred += predictions.tolist()
          fnames += fname
          #print(fname)
     ff = open(exp_dir + '/eval' ,'a')
     assert len(fnames) == len(y_pred)
     for (f, yp, yt) in list(zip(fnames, y_pred, y_true)):
          if yp == yt:
            continue
          ff.write( f + ' ' + str(yp) + ' ' + str(yt) + '\n')
     ff.close()

     averaged_loss = running_loss / (len(val_loader))
     recall = get_metrics(y_pred, y_true)
     log_value("Unweighted Recall per epoch", recall, global_epoch)
     log_value("validation loss (per epoch)", averaged_loss, global_epoch)
     print("Validation Loss: ", averaged_loss)
     print("Unweighted Recall for the validation set:  ", recall)
     print('\n')
     return recall, model.train()

def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0):
    if use_cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    global global_step, global_epoch

    # https://github.com/r9y9/wavenet_vocoder/blob/c4c148792c6263afbedb9f6bf11cd552668e26cb/train.py#L793
    if hparams.exponential_moving_average is not None:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
	                ema.register(name, param.data)
    else:
        ema = None

    while global_epoch < nepochs:
        model.train()
        h = open(logfile_name, 'a')
        running_loss = 0. # mfcc, lengths, mel, mol, lid, fnames
        running_loss_vq = 0.
        running_loss_encoder = 0.
        running_entropy = 0.
        running_loss_lid = 0.

        for step, (mfcc, mfcc_lengths, mel, mol, lid, fnames) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                mfcc_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            mfcc = mfcc[indices]
            mel = mel[indices]
            mol = mol[indices]
            lid = lid[indices]

            #print(fnames, indices)
            #sys.exit()

            # Feed data
            mfcc, mel, mol, lid = Variable(mfcc), Variable(mel), Variable(mol), Variable(lid)
            if use_cuda:
                mfcc, mel, mol, lid = mfcc.cuda(), mel.cuda(), mol.cuda(), lid.cuda().long()

            logits, vq_penalty, encoder_penalty, entropy = model(mfcc, mel, mol, mfcc_lengths=sorted_lengths)

            # Loss
            #print("Shape of logits and lid: ", logits.shape, lid.shape)
            lid_loss = criterion(logits, lid)
            encoder_weight = 0.01 * min(1, max(0.1, global_step / 1000 - 1)) # https://github.com/mkotha/W$
            loss = lid_loss + vq_penalty + encoder_penalty * encoder_weight

            #print(loss)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()
            model.quantizer.after_update()

            if ema is not None:
              for name, param in model.named_parameters():
                 if name in ema.shadow:
                    ema.update(name, param.data)


            if global_step % checkpoint_interval == 0:

               save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch, ema=ema)

            # Logs
            log_value("Training Loss", float(loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            log_value("learning rate", current_lr, global_step)
            log_value("VQ Penalty", vq_penalty, global_step)
            log_value("Encoder Penalty", encoder_penalty, global_step)
            log_value("Entropy", entropy, global_step)

            global_step += 1
            running_loss += loss.item()
            running_loss_vq += vq_penalty.item()
            running_loss_encoder += encoder_penalty.item()
            running_entropy += entropy
            running_loss_lid += lid_loss.item()

            #print(loss.item())

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) 
                + " LID Loss: " + format(running_loss_lid / (len(train_loader)))
                + " VQ Penalty: " + format(running_loss_vq / (len(train_loader))) 
                + " Encoder Penalty: " + format(running_loss_encoder / (len(train_loader))) 
                + " Entropy: " + format(running_entropy / (len(train_loader)))
                + '\n')
        h.close()
        recall, model = validate_model(model, val_loader)
        log_value("Unweighted Recall per epoch", recall, global_epoch)
        global_epoch += 1

    return model, ema


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

    # Vocab size
    with open(vox_dir + '/' + 'etc/ids_latents.json') as  f:
       latent_ids = json.load(f)

    latent_ids = dict(latent_ids)
    print(latent_ids)

    idsdict_file = checkpoint_dir + '/ids_latents.json'

    with open(idsdict_file, 'w') as outfile:
       json.dump(latent_ids, outfile)

    feats_name = 'mfcc'
    mfcc_train = float_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    mfcc_val = float_datasource( vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)

    feats_name = 'lid'
    X_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    X_val = categorical_datasource(vox_dir + '/' +  'fnames.val', vox_dir + '/' +  'etc/falcon_feats.desc', feats_name,  vox_dir + '/' +  'festival/falcon_' + feats_name)

    feats_name = 'fnames'
    fnames_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    fnames_val = categorical_datasource( vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)

    feats_name = 'r9y9inputmol'
    mol_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)
    mol_val = categorical_datasource( vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name)

    feats_name = 'r9y9outputmel'
    mel_train = float_datasource(vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    mel_val = float_datasource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    # Dataset and Dataloader setup X, mfcc, mel, mol, fnames)
    trainset =  LIDmfccmelmolDataset(X_train, mfcc_train, mel_train, mol_train, fnames_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lidmfccmelmol, pin_memory=hparams.pin_memory)

    valset =  LIDmfccmelmolDataset(X_val, mfcc_val, mel_val, mol_val, fnames_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lidmfccmelmol, pin_memory=hparams.pin_memory)

    # Model
    model = LIDmfccmelmol2(39, 80)
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
        model, ema = train(model, train_loader, val_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh)
        model = clone_as_averaged_model(model, ema)
        recall, model = validate_model(model, val_loader)
        print("Final Recall: ", recall)
        recall, model = validate_model(model, val_loader)
        print("Final Recall: ", recall)
        recall, model = validate_model(model, val_loader)
        print("Final Recall: ", recall)
 

    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)


