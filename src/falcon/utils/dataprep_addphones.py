import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
import json

'''Syntax
python3.5 $FALCONDIR/dataprep_addphones.py vox.ehmm/etc/txt.phseq.data

X -> typically this file is vox/ehmm/etc/txt.phseq.data but can be anything else as well
It is a file with the filename and phone sequence in following format:

filename1 ph1 ph2 ph3 ...
filename2 ph1 ph2 ph3 ...
filename3 ph1 ph2 ph3 ...

Y -> this is typically 'vox' but can be anything else
basically the script creates a directory vox/festival/falcon_phones and puts phones corresponding to each prompt as a separate file. 
the script also creates a dictionary of phones so that we can use them in deep learning models. The name of this is ids_phones.json and it is placed in vox/etc/
It also updates a file called vox/etc/falcon_feats.desc. This is a description file that models use later.

'''

phseq_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_phones'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_phones.json'
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

    fname = str(line.split()[0])
    text = '< ' + ' '.join(k for k in line.split()[1:]) + ' >'
    print(text)
    text_ints = ' '.join(str(ids_dict[k]) for k in text.split())

    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(text + '\n')
    g.close()



g = open(desc_file, 'a')
g.write('phones|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)

