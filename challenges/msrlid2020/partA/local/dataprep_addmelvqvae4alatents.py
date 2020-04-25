import os, sys
FALCON_DIR= os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import numpy as np
from utils import audio
from utils.misc import *
import re
from torch.autograd import Variable
from model import MelVQVAEv4

'''Syntax
python3.5 $FALCONDIR/dataprep_addmspec.py etc/tdd .
'''

tdd_file  = sys.argv[1]
vox_dir = sys.argv[2]
checkpoint = sys.argv[3]

melfeats_dir = vox_dir + '/festival/falcon_r9y9outputmel'
latents_dir = vox_dir + '/festival/falcon_melvqvae4alatents'
assure_path_exists(latents_dir)
assure_path_exists(melfeats_dir)
desc_file = vox_dir + '/etc/falcon_feats.desc'
latents_desc_file = vox_dir + '/etc/latents_melvqvae4a.desc'

latents_dict = defaultdict(lambda: len(latents_dict))
latentsdict_file = vox_dir + '/etc/ids_latents.json'
#latents_dict[0]

model = MelVQVAEv4(n_vocab=257,
                 embedding_dim=256,
                 mel_dim=hparams.num_mels,
                 linear_dim=hparams.num_freq,
                 r=hparams.outputs_per_step,
                 padding_idx=hparams.padding_idx,
                 use_memory_mask=hparams.use_memory_mask,
                 )
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint["state_dict"])

if torch.cuda.is_available():
   model = model.cuda()
model.eval()


def get_latents(sequence):

    sequence = np.array(mel)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    sequence = sequence.cuda()
    with torch.no_grad():
       latents, entropy = model.quantizer.get_quantizedindices(sequence.unsqueeze(2)) 
       return latents, entropy



f = open(tdd_file, encoding='utf-8')
h = open(latents_desc_file, 'w')
ctr = 0
for line in f:
 if len(line) > 2:

    ctr += 1
    line = line.split('\n')[0]
    fname = line.split()[0]

    if ctr % 100 == 1:
       print("Processed ", ctr, "lines")

    fname = line.split()[0]
    mel_fname = melfeats_dir + '/' + fname + '.feats.npy'
    mel = np.load(mel_fname)

    # Get latents
    latents,entropy = get_latents(mel)

    latents_fname = latents_dir + '/' + fname + '.feats'
    np.save(latents_fname, latents, allow_pickle=False)

    h.write(fname + ' ' + str(entropy) + '\n')

    latent_ints = ' '.join(str(latents_dict[k]) for k in latents)


f.close()
h.close()
g = open(desc_file, 'a')
g.write('melvqvae4alatents|single|categorical' + '\n')
g.close()

with open(latentsdict_file, 'w') as outfile:
  json.dump(latents_dict, outfile)


