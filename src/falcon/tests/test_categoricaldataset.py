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

feats_name = 'phones'
phones_dataset_1 = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_dict = phones_dataset_1.get_dict()
phones_dataset_2 = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name, feats_dict)

dataset = CombinedDataset([phones_dataset_1, phones_dataset_2])
dataloader = DataLoader(dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=4, 
                          collate_fn=collate_fn_combined
                          )

for data in dataloader:
    print("Here ")
print(data[0], data[1])
