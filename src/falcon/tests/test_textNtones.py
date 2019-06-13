import sys, os
from os.path import dirname, join
import json

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get("FALCONDIR")
sys.path.append(FALCON_DIR)
##############################################


from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from utils.misc import *


charids = make_charids_text( 'etc/txt.done.data.tacotron')
charids['UNK']
charids['SPACE']
charids = update_charids(charids, 'scripts/txt.done.data.tones')
print(charids)

'''Old
feats_name = 'text'
X = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name, charids)
feats_name = 'lspec'
Y = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_name = 'mspec'
Mel = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_name = 'tones'
tones = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name, charids)
print(dict(X.get_dict()))
'''

feats_name = 'textNtones'
X = CategoricalDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name, charids)
feats_name = 'lspec'
Y = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)
feats_name = 'mspec'
Mel = FloatDataSource('fnames.train', 'etc/falcon_feats.desc', feats_name, 'festival/falcon_' + feats_name)

dataset = PyTorchCombinedDataset(X, Mel, Y)
dataloader = DataLoader(dataset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=1,
                          collate_fn=collate_fn_textNtones
                          )

id2char = {v:k for (k,v) in charids.items()}

for (x, tones, input_lengths, mel, y) in dataloader:

    #print("Shapes of x and tones: ", x.shape, tones.shape, x, tones)
    text = x[0,:].cpu().numpy()
    tone = tones[0,:].cpu().numpy()
    print(''.join(id2char[k] for k in text))
    print(' '.join(id2char[k] for k in tone))
    assert len(text) == len(tone)
    sys.exit()


