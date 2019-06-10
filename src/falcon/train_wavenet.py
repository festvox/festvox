import os, sys
import time

import numpy as np
from util import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import soundfile as sf

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ['FALCONDIR']
sys.path.append(FALCON_DIR)
##############################################

from utils.misc import *
from models import Wavenet_Barebones
from hparams_arctic import hparams_wavenet as hparams

### Flags
max_timesteps = 8192
frame_period = 256
max_epochs = 100
max_frames_test = 100

data_dir = 'wavenetdataprepdir'
tdd_file_train = data_dir + '/all.txt.train'
tdd_file_test = data_dir + '/all.txt.test'
logfile_name = 'log_wavenet'
g = open(logfile_name, 'w')
g.close()
checkpoint_dir = 'checkpoints_wavenet'

def test(dataloader, partial_flag = 1):
  model.eval()
  for idx, (mel, wav) in enumerate(dataloader):
      #print("Shape of wav: ", wav.shape)
      orig_mulaw = inv_mulaw_quantize(np.transpose(wav))
      #print("Shape of wav: ", orig_mulaw.shape)
      sf.write('wav_predictions/' + 'test_original_step' + str(updates) + '.wav', np.asarray(orig_mulaw), 16000,format='wav',subtype="PCM_16")
      wav, mel = wav.cuda(), mel[:, 0:max_frames_test].cuda() 
      wav_pred = model.forward_incremental(mel).squeeze(1)
      x_hat = torch.max(wav_pred,-1)[1]
      #print("Shape of prediction: ", x_hat.shape)
      x_hat = x_hat.detach().cpu().numpy()
      x_hat = inv_mulaw_quantize(x_hat)
      sf.write('wav_predictions/' + 'test_predicted_step' + str(updates) + '.wav', np.asarray(x_hat), 16000,format='wav',subtype="PCM_16")
      if partial_flag:
         return

def train(dataloader):
  training_loss = 0
  global updates
  start_time = time.time()
  for idx, (mel, wav) in enumerate(dataloader):
      updates += 1
      mel, wav = mel.cuda(), wav.cuda()
      wav_pred = model.forward_convupsampling(wav, mel)
      wav = wav[:, 1:]
      loss = criterion(wav_pred.contiguous().view(wav_pred.shape[0]*wav_pred.shape[1], -1), wav.contiguous().view(-1))
      if updates % 1000 == 1:
        print("Training loss after ", updates, " updates: ", loss.item(), " and it took ", idx/(time.time() - start_time), " updates per second")
        g = open(logfile_name, 'a')
        g.write('Training loss after ' + str(updates) + ' updates ' + str(loss.item()) + " and it took " + str(idx/(time.time() - start_time)) + ' updates per second ' + '\n')
        g.close()
      if updates % 10000 == 1: 
        test(test_loader)

      training_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
      optimizer.step()

  return training_loss/idx 


trainset = load_tdd(tdd_file_train, data_dir)
train_loader  = DataLoader(trainset,
                          batch_size=2,
                          shuffle=True,
                          num_workers=4,
                          collate_fn = collate_fn_wavenet
                          )
testset = load_tdd(tdd_file_test, data_dir)
test_loader  = DataLoader(testset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4
                          )

model = Wavenet_Barebones()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
        hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay,
        amsgrad=hparams.amsgrad)

updates = 0
test(test_loader)
for epoch in range(max_epochs):
   train(train_loader)
   #test(test_loader)
