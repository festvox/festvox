""" Prepare phones N qF0s as features 

Usage: local/dataprep_addphonesNqF0s.py <tdd_file> <vox_dir> 

options:
    -h, --help               Show help message.

"""
from docopt import docopt

import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
from collections import defaultdict


### Flags
generate_feats_flag = 0


args = docopt(__doc__)
tdd_file  = args['<tdd_file>']
vox_dir = args['<vox_dir>']
feats_dir = vox_dir + '/festival/falcon_qF0'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
ids_file = vox_dir + '/etc/ids_qF0s.json'

def get_interpolated_tones(text, tones):
    print(text)
    print(tones)
    #sys.exit()
    print("The length of text and tones: ", len(text), len(tones))
    assert len(text.split()) == len(tones.split())
    interpolated_tones = []
    chars = []
    for (phone, tone) in list(zip(text.split(), tones.split())):
        #word = re.sub(r'[^\w\s]','',word)
        #for char in word:
        interpolated_tones.append(tone )
        #interpolated_tones.append('SPACE_SPACE' )
    return interpolated_tones


f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    line = line.split('\n')[0]
    fname = line.split()[0].split('.')[0]
    phones_qF0s = ' '.join(k for k in line.split()[1:])
    phones = ' '.join(k.split('_')[0] for k in phones_qF0s.split())
    qF0s = ' '.join(k.split('_')[1] for k in phones_qF0s.split())
    qF0s_interpolated = get_interpolated_tones(phones, qF0s)
    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    try:
      g = open(feats_dir + '/' + fname + '.feats', 'w')
      g.write(' '.join(k for k in qF0s_interpolated) +'\n')
      g.close()
    except UnicodeEncodeError:
      print(line)
      sys.exit()

g = open(desc_file, 'a')
g.write('qF0|single|categorical' + '\n')
g.close()

