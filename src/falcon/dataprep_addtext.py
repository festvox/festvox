import os, sys
FALCON_DIR= '/home/srallaba/projects/text2speech/repos/festvox/src/falcon/'
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
feats_dir = vox_dir + '/festival/falcon_text'
assure_path_exists(feats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
idsdict_file = vox_dir + '/etc/ids.json'
ids_dict = defaultdict(lambda: len(ids_dict))
ids_dict['>']
ids_dict['UNK']
ids_dict['<']
ids_dict[' ']
ids_dict[',']

f = open(tdd_file, encoding='utf-8')
ctr = 0
for line in f:
 if len(line) > 2:
    ctr += 1
    fname = line.split('\n')[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    text = re.sub(r'[^\w\s]','', ' '.join(k for k in line.split()[1:])).strip().split()

    fname = line.split()[0]
    text = '< ' + ' '.join(k.lower() for k in text) + ' >'
    #print(text)
    text_ints = ' '.join(str(ids_dict[k.lower()]) for k in text)
    ### This is not a good fix - Sai Krishna 27 May 2019 #########
    # https://stackoverflow.com/questions/9942594/unicodeencodeerror-ascii-codec-cant-encode-character-u-xa0-in-position-20 
    text = text.encode('ascii', 'ignore').decode('ascii')
    ##############################################################


    g = open(feats_dir + '/' + fname + '.feats', 'w')
    g.write(text + '\n')
    g.close()



g = open(desc_file, 'a')
g.write('text|single|categorical' + '\n')
g.close()

with open(idsdict_file, 'w') as outfile:
  json.dump(ids_dict, outfile)


print(ids_dict)
