import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
import json
from arabic_pronounce import phonetise

'''Syntax
python3.5 $FALCONDIR/dataprep_addphones.py temp_filtered.csv vox

'''

filtered_file  = sys.argv[1]
vox_dir = sys.argv[2]
feats_dir = vox_dir + '/festival/falcon_phones'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids_phones.json'
ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['EOS']
ids_dict['SOS']
k = open(vox_dir +'/ehmm/etc/txt.phseq.data', 'w')

def _maybe_get_arpabet(word):
    pronunciations = phonetise(word)
    toBeReturned = pronunciations[0] if len(pronunciations)==1 else pronunciations[1]
    return toBeReturned


def get_phones(text):
    arr = []
    for word in text.split(' '):
      if word in [" ", ""]:
        pass
      elif word in [",", '.', '-', "SOS", "EOS"]:
        x = word
        arr.append(x)
      else:
        x = _maybe_get_arpabet(word)
        arr.append(x)
    text = ' '.join(arr)
    return text

f = open(filtered_file, encoding='utf-8')
h = open(vox_dir + '/fnames', 'w')
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    line = line.split('\n')[0]
    fname = line.split(',')[0]
    text = ' '.join(k for k in line.split(',')[1:])

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    text = 'SOS ' + text + ' EOS'
    text = get_phones(text)
    print(text)
    k.write(fname + ' ' + text + '\n')
    text_ints = ' '.join(str(ids_dict[k]) for k in text.split())

    g = open(feats_dir + '/ARA_NORM_' + fname + '.feats', 'w')
    g.write(text + '\n')
    g.close()
    
    h.write('ARA_NORM_' + fname + '\n')


h.close()
k.close()
g = open(desc_file, 'a')
g.write('phones|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)

