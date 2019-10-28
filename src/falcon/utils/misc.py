from os.path import join, expanduser
from collections import defaultdict
import numpy as np
import torch
from .audio import *
from .plot import *
import sys,os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Made the directory ", path) 

def get_padded_1d(seq):
    lengths = [len(x) for x in seq]
    max_len = np.max(lengths)
    return _pad(seq, max_len)
 

def _pad(seq, max_len):
    #print("Shape of seq: ", seq.shape, " and the max length: ", max_len)     
    assert len(seq) < max_len
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{}.pth".format(step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    # idx = np.random.randint(0, len(input_lengths))
    idx = 0
    #input_length = input_lengths[idx]

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(
        global_step))
    # alignment = attn[idx].cpu().data.numpy()[:, :input_length]
    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment, global_step)

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{}_predicted.wav".format(
        global_step))
    save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)


def learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr

def get_fnames(fnames_file):
    filenames_array = []
    f = open(fnames_file)
    for line in f:
      line = line.split('\n')[0]
      filenames_array.append(line)
    return filenames_array

def populate_featdict(desc_file):
    f = open(desc_file)
    dict = {}
    dict_featnames = {}
    idx = 0
    for line in f:
       line = line.split('\n')[0] 
       name, length, type = line.split('|')[0], line.split('|')[1], line.split('|')[2]
       dict[name + '_length'] = length
       dict[name + '_type'] = type
       dict_featnames[idx] = name
       idx += 1
    return dict, dict_featnames


def get_featmetainfo(desc_file, feat_name):

    f = open(desc_file)
    for line in f:
        line = line.split('\n')[0]
        feat = line.split('|')[0]
        if feat_name == feat:
           feat_length, feat_type = line.split('|')[1], line.split('|')[2]
           return feat_length,feat_type 

def populate_featarray(fname, feats_dir, feats_dict):
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0]
        feats  = line.split()
        for feat in feats:
            feats_array.append(feats_dict[feat])
    feats_array = np.array(feats_array)
    return feats_array

def populate_textarray(fname, feats_dir, feats_dict):
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0]
        feats  = line
        for feat in feats:
            feats_array.append(feats_dict[feat])
    feats_array = np.array(feats_array)
    return feats_array

def populate_phonesarray(fname, feats_dir, feats_dict):
    #print("Fname: ", fname)
    #print("Feats dir: ", feats_dir)
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0]
        feats  = line.split()[1:]
        for feat in feats:
            feats_array.append(feats_dict[feat])
    feats_array = np.array(feats_array)
    return feats_array

class FileNameDataSource(Dataset):

    def __init__(self, fnames_file):
       self.fnames_file = fnames_file
       f = open(self.fnames_file)
       self.fnames_array = []
       for line in f:
          line = line.split('\n')[0]
          self.fnames_array.append(line)

    def __getitem__(self, idx):
       return self.fnames_array[idx]

    def __len__(self):
       return len(self.fnames_array)


class CategoricalDataSource(Dataset):
    '''Syntax
    dataset = CategoricalDataSource(fnames.txt.train, etc/falcon_feats.desc, feat_name, feats_dir)

    '''

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
      self.fnames_file = fnames_file
      print("Fnames file is ", self.fnames_file)
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.filenames_array = get_fnames(self.fnames_file)
      print("Feat name is ", feat_name)
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)
      self.feats_dir = feats_dir
      self.feats_dict = defaultdict(lambda: len(self.feats_dict)) if feats_dict is None else feats_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'text':
            return populate_textarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'indiantext':
            return populate_indiantextarray(fname, self.feats_dir, self.feats_dict)
        feats_array = populate_featarray(fname, self.feats_dir, self.feats_dict) 
        return feats_array

    def __len__(self):
        return len(self.filenames_array)


class FloatDataSource(Dataset):
    '''Syntax
    dataset = FloatDataSource(fnames.txt.train, etc/falcon_feats.desc, feat_name, feats_dir)

    '''

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
      self.fnames_file = fnames_file
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.filenames_array = get_fnames(self.fnames_file)
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)
      self.feats_dir = feats_dir
      self.feats_dict = defaultdict(lambda: len(self.feats_dict)) if feats_dict is None else feats_dict

    def __getitem__(self, idx):

        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
        feats_array = np.load(fname)
        return feats_array

    def __len__(self):
        return len(self.filenames_array)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, y_batch

class DataParallelFix(torch.nn.DataParallel):

    """
    Temporary workaround for https://github.com/pytorch/pytorch/issues/15716.
    """

    def __init__(self,  *args, **kwargs):
        super().__init__( *args, **kwargs)

        self._replicas = None
        self._outputs = None

    def forward(self, *inputs, **kwargs):
        if not self.device_ids or len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        print("Length of inputs: ", len(inputs))

        self._replicas = self.replicate(self.module,
                                  self.device_ids[:len(inputs)])
        self._outputs = self.parallel_apply(self._replicas, inputs, kwargs)

        return self.gather(self._outputs, self.output_device)



def data_parallel_workaround(model, input):
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    return y_hat, outputs, replicas
