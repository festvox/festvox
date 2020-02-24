import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment

from sklearn.manifold import TSNE
import json

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)


class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

class categorical_datasource_spk(categorical_datasource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict, spk_dict):

        super(categorical_datasource_spk, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

        self.spk_dict = spk_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        if self.feat_name == 'speaker':
            speaker = fname.split('_')[0]
            return self.spk_dict[speaker]

    def get_spkdict(self):
        return self.spk_dict


### Collate Stuff
def _pad(seq, max_len):
    #print("Shape of seq: ", seq.shape, " and the max length: ", max_len)     
    #assert len(seq) < max_len
    if len(seq) == max_len:
        return seq
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


# We get (x_array, spk_id), y, mel
def collate_fn_vqvae(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    mels =  [x[1] for x in batch]
    linears = [x[2] for x in batch]
    spk_ids =  [x[0] for x in batch]

    input_lengths = [len(mel) for mel in mels]
    mel_lengths = [len(mel) for mel in mels]
    if np.all(mel_lengths) is False:
       print("Check this ", mel_lengths)
       sys.exit()

    max_input_len = np.max(input_lengths)

    max_target_len = np.max([len(x) for x in mels]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    spk_batch = torch.LongTensor(spk_ids)

    input_lengths = torch.LongTensor(input_lengths)
    b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(linear, max_target_len) for linear in linears],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    return spk_batch, input_lengths, mel_batch, y_batch


# We get (x_array, spk_id), y, mel
def collate_fn_vqvaemaxlen(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    max_allowed_len = 256
    if max_allowed_len % r != 0:
        max_allowed_len += r - max_allowed_len % r
        assert max_allowed_len % r == 0

    mels =  [x[1] for x in batch]
    linears = [x[2] for x in batch]
    spk_ids =  [x[0] for x in batch]

    mel_batch = []
    y_batch = []
    for mel,y in list(zip(mels, linears)):
        if mel.shape[0] < max_allowed_len :
           mel = _pad_2d(mel, max_allowed_len )
           y = _pad_2d(y, max_allowed_len)
        elif mel.shape[0] == max_allowed_len :
           pass
        else:
           start_index = np.random.randint(0, mel.shape[0]-max_allowed_len)
           mel = mel[start_index:start_index + max_allowed_len]
           y = y[start_index:start_index + max_allowed_len]
        mel_batch.append(mel)
        y_batch.append(y)

    mel_batch = np.array(mel_batch)
    y_batch = np.array(y_batch)
    spk_batch = torch.LongTensor(spk_ids)
    mel_batch = torch.FloatTensor(torch.from_numpy(mel_batch))
    y_batch = torch.FloatTensor(torch.from_numpy(y_batch))

    return spk_batch, mel_batch, y_batch


def visualize_latent_embeddings(model, checkpoints_dir, step):
    print("Computing TSNE")
    latent_embedding = model.quantizer.embedding0.squeeze(0).detach().cpu().numpy()
    num_classes = model.num_classes

    #ppl_array = [5, 10, 40, 100, 200]
    ppl_array = [40]
    for ppl in ppl_array:

       embedding = TSNE(n_components=2, verbose=1, perplexity=ppl).fit_transform(latent_embedding)

       y = embedding[:,0]
       z = embedding[:,1]

       fig, ax = plt.subplots()
       ax.scatter(y, z)

       for i  in range(num_classes):
          ax.annotate(i, (y[i], z[i]))

       path = checkpoints_dir + '/step' + str(step) + '_latent_embedding_perplexity_' + str(ppl) + '.png'
       plt.tight_layout()
       plt.savefig(path, format="png")
       plt.close()

