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

feats_name = 'lspec'
dataset = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
dataloader = DataLoader(dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4,
                          collate_fn=collate_fn_float
                          )

for data in dataloader:
    print("Here ")
print(data[0], data[1])
