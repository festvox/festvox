import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re


'''Syntax
python3.5 $FALCONDIR/dataprep_addtones.py scripts/txt.done.data.interpolatedtones .

'''

### Flags
generate_feats_flag = 0

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_tones'

assure_path_exists(feats_dir)

desc_file = vox_dir + '/etc/falcon_feats.desc'

f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    line = line.split('\n')[0]

    fname = line.split('|')[0].split('.')[0]
    tones = ''.join(k for k in line.split('|')[1:])
    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(tones + '\n')
    g.close()



g = open(desc_file, 'a')
g.write('tones|single|categorical' + '\n')
g.close()

