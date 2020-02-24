import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
import json

'''Syntax
python3.5 $FALCONDIR/dataprep_addtext.py etc/tdd .

'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_mask'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_mask.json'
ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['>']
ids_dict['<']

f = open(tdd_file)
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]
    if ctr == 1:
       continue
    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split(',')[0].split('.wav')[0]
    mask = line.split(',')[1].split('\n')[0]
    print(mask)
    if mask == 'clear':
       mask = 0
    elif mask == 'mask':
       mask = 1
    elif mask == '?':
       mask='?'
    else:
       print("Something is wrong with ", mask)
       sys,exit()

    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(str(mask) + '\n')
    g.close()



g = open(desc_file, 'a')
g.write('mask|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)

