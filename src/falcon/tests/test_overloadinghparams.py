import sys, os
from os.path import dirname, join
import json
from torch.utils import data as data_utils
import re

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get("FALCONDIR")
sys.path.append(FALCON_DIR)
##############################################


from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from utils.misc import *





print(hparams)
with open('preset_test.json') as f:
   hparams.parse_json(f.read())
print(hparams)

