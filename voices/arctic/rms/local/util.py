import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment


### Text Processign Stuff
def populate_phonesarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary") 
       sys.exit()

    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats

def populate_phonesNstressarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary")
       sys.exit()
    f = open(fname)
    arr = {}
    arr['fname'] = fname
    for line in f:
        line = line.split('\n')[0].split()
        phones  = [feats_dict[phdur.split('_')[0]] for phdur in line]
        stress  = [int(float(phdur.split('_')[1])) for phdur in line]

    phones = np.array(phones)
    stress = np.array(stress)
    arr['phones'] = phones
    arr['stress'] = stress 
    return arr

### Data Source Stuff
class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'phonesnossil':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        elif self.feat_name == 'phonesNstress':
            return populate_phonesNstressarray(fname, self.feats_dir, self.feats_dict)

        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)


### Collate stuff

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



def collate_fn_phonesNqF0s(batch):


    r = hparams.outputs_per_step
    input_lengths = [len(x[0]['phones']) for x in batch]

    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    x_inputs = [_pad(x[0]['phones'], max_input_len) for x in batch]
    x_batch = torch.LongTensor(x_inputs)

    x_qF0s = [_pad(x[0]['qF0s'], max_input_len) for x in batch]
    x_qF0s_batch = torch.LongTensor(x_qF0s)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, input_lengths, x_qF0s_batch, mel_batch, y_batch



def collate_fn_phonesNstress(batch):


    r = hparams.outputs_per_step
    input_lengths = [len(x[0]['phones']) for x in batch]

    max_input_len = np.max(input_lengths) + 1
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    x_inputs = [_pad(x[0]['phones'], max_input_len) for x in batch]
    x_batch = torch.LongTensor(x_inputs)

    x_qF0s = [_pad(x[0]['stress'], max_input_len) for x in batch]
    x_qF0s_batch = torch.LongTensor(x_qF0s)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, input_lengths, x_qF0s_batch, mel_batch, y_batch



#### Visualization Stuff

def visualize_phone_embeddings(model, checkpoints_dir, step):

    print("Computing TSNE")
    phone_embedding = model.embedding
    phone_embedding = list(phone_embedding.parameters())[0].cpu().detach().numpy()
    phone_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(phone_embedding)

    with open(checkpoints_dir + '/ids_phones.json') as  f:
       phones_dict = json.load(f)

    ids2phones = {v:k for (k,v) in phones_dict.items()}
    phones = list(phones_dict.keys())
    y = phone_embedding[:,0]
    z = phone_embedding[:,1]

    fig, ax = plt.subplots()
    ax.scatter(y, z)

    for i, phone in enumerate(phones):
        ax.annotate(phone, (y[i], z[i]))

    path = checkpoints_dir + '/step' + str(step) + '_embedding_phones.png'
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

### Misc
def visualize_latent_embeddings(model, checkpoints_dir, step):
    return
    print("Computing TSNE")
    latent_embedding = model.quantizer.embedding0.squeeze(0).detach().cpu().numpy()
    num_classes = model.num_classes

    ppl_array = [5, 10, 40, 100, 200]
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

