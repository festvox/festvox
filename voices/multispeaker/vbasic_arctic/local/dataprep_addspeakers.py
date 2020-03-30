import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
import json
    
'''Syntax
python3.5 $FALCONDIR/dataprep_addspeakers.py vox//ehmm/etc/txt.phseq.data vox

'''   
    
phseq_file  = sys.argv[1]
vox_dir = sys.argv[2]
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_speakers.json'
ids_dict = defaultdict(lambda: len(ids_dict))

g = open(vox_dir + '/etc/fnamesNspeaker', 'w')

f = open(phseq_file)
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = str(line.split()[0])
    spk = fname.split('_')[0]

    g.write(fname + ' ' + str(ids_dict[spk]) + '\n')


g.close()
g = open(desc_file, 'a')
g.write('speaker|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)

