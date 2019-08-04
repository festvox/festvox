import sys, os
from os.path import dirname, join
### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get("FALCONDIR")
sys.path.append(FALCON_DIR)
##############################################


from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from utils.misc import *
from torch.utils import data as data_utils

import json

charids_file = 'etc/ids_phones.json'
with open(charids_file) as f:
   char_ids = json.load(f)
print(char_ids)
ids2chars = {v:k for (k,v) in char_ids.items()}

fnames_array = []
f = open('fnames.train')
for line in f:
   line = line.split('\n')[0]
   fnames_array.append(line)
f.close()
feats_name = 'phones'
X = CategoricalDataSource_fnames_tones(fnames_array, 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name,  char_ids)

feats_name = 'lspec'
Y_train = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)

feats_name = 'mspec'
Mel_train = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)

dataset = PyTorchDataset(X, Mel_train, Y_train)
data_loader = data_utils.DataLoader(
        dataset, batch_size=1,
        num_workers=0, shuffle=False,
        pin_memory=hparams.pin_memory,
        collate_fn=collate_fn)


for (x,l,m, y) in data_loader:
    x = x.squeeze(0).numpy()
    print(x.shape)
    print("Here ", ' '.join(str(ids2chars[k]) for k in x), m.shape)

