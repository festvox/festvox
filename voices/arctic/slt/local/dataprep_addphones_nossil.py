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

phseq_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_phonesnossil'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_phonesnossil.json'
ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['>']
ids_dict['<']

f = open(phseq_file)
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split()[0]
    text = '< ' + ' '.join(k for k in line.split()[1:]) + ' >'
    print(text)
    text = text.replace(' ssil ', ' ')
    print(text)
    text_ints = ' '.join(str(ids_dict[k]) for k in text.split())
    
    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(text + '\n')
    g.close()



g = open(desc_file, 'a')
g.write('phonesnossil|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)

