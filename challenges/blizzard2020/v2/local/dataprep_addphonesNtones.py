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

python3.5 local/dataprep_addphonesNtones.py ../../../../../voices/v2/tdd.phonesNtones  vox/
'''

phseq_file  = sys.argv[1]
vox_dir = sys.argv[2]
phones_dir = vox_dir + '/festival/falcon_phones'
tones_dir = vox_dir + '/festival/falcon_tones'
assure_path_exists(phones_dir)
assure_path_exists(tones_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_phones.json'
ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['>']
ids_dict['<']
tonesdict_file = vox_dir + '/etc/ids_tones.json'
tones_dict = defaultdict(lambda: len(tones_dict))

f = open(phseq_file)
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = str(line.split()[0])
    text = '<_0 ' + ' '.join(k for k in line.split()[1:]) + ' >_0'
    print(text)
    text_ints = ' '.join(str(k.split('_')[0]) for k in text.split())
    tone_ints = ' '.join(str(k.split('_')[1]) for k in text.split())

    g = open(phones_dir + '/blizzard_' + fname + '.feats', 'w')
    g.write(text_ints + '\n')
    g.close()

    g = open(tones_dir + '/blizzard_' + fname + '.feats', 'w')
    g.write(tone_ints + '\n')
    g.close()

    text_ints = ' '.join(str(ids_dict[k.split('_')[0]]) for k in text.split())
    tone_ints = ' '.join(str(tones_dict[k.split('_')[1]]) for k in text.split())
    

g = open(desc_file, 'a')
g.write('phones|single|categorical' + '\n')
g.write('tones|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)

with open(tonesdict_file, 'w') as outfile:
  json.dump(tones_dict, outfile)

print(ids_dict)

