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

feats_name = 'indiantext'
X = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_name = 'lspec'
Y_train = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_name = 'mspec'
Mel_train = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)

dataset = PyTorchDataset(X, Mel_train, Y_train)
data_loader = data_utils.DataLoader(
        dataset, batch_size=4,
        num_workers=4, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)


for data in data_loader:
    print("Here ", data[2].shape)
