import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment



### Text Processing Stuff
def populate_phonesarray(fname, feats_dict):
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


def get_fnames(fnames_file, scp_file):
    scp_dict = {}
    f = open(scp_file)
    for line in f:
      line = line.split('\n')[0]
      fname, location = line.split()[0], line.split()[1]
      scp_dict[fname] = location

    filenames_array = []
    f = open(fnames_file)
    for line in f:
      line = line.split('\n')[0]
      #print(line)
      filenames_array.append(scp_dict[line])
    return filenames_array


### Data Source Stuff
class categorical_datasource(Dataset):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dict=None, spk_dict=None):

      self.fnames_file = fnames_file
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.vox_dir = 'vox'
      self.filenames_array = get_fnames(self.fnames_file, self.vox_dir + '/etc/fnamesN' + self.feat_name)
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)
      self.feats_dict = defaultdict(lambda: len(self.feats_dict)) if feats_dict is None else feats_dict
      self.spk_dict = defaultdict(lambda: len(self.spk_dict)) if spk_dict is None else spk_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        #print("Fname is ", fname)
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dict)
        elif self.feat_name == 'speaker' :
            return int(fname)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

    def __len__(self):
       return len(self.filenames_array)

    def get_spkdict(self):
        return self.spk_dict
 
class float_datasource(Dataset):

    def __init__(self, fnames_file, desc_file, feat_name):

      self.fnames_file = fnames_file
      self.feat_name = feat_name
      self.desc_file = desc_file
      self.vox_dir = 'vox'
      self.filenames_array = get_fnames(self.fnames_file, self.vox_dir + '/etc/fnamesN' + self.feat_name)
      self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)

    def __getitem__(self, idx):

        fname = self.filenames_array[idx]
        return np.load(fname)

    def __len__(self):
       return len(self.filenames_array)


class MultispeakerDataset(object):
    def __init__(self, X, spk, Mel, Y):
        self.X = X
        self.spk = spk
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.spk[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


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
def collate_fn_spk(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    phones = [x for (x,_,_,_) in batch]
    mels =  [x for (_, _, x,_) in batch]
    linears = [x for (_,_,_, x) in batch]
    spk_ids =  [x for (_,x,_,_) in batch]

    input_lengths = [len(p) for p in phones]
    mel_lengths = [len(mel) for mel in mels]
    if np.all(mel_lengths) is False:
       print("Check this ", mel_lengths)
       sys.exit()

    max_input_len = np.max(input_lengths)

    max_target_len = np.max([len(x) for x in mels]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0
    try:
      a = np.array([_pad(p, max_input_len) for p in phones], dtype=np.int)
    except Exception as e:
      print("Exception here : ", chars)
      print(e)
      sys.exit()
    x_batch = torch.LongTensor(a)
    #print(spk_ids)
    spk_batch = torch.LongTensor(spk_ids)

    input_lengths = torch.LongTensor(input_lengths)
    b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(linear, max_target_len) for linear in linears],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    return x_batch, spk_batch, input_lengths, mel_batch, y_batch



def visualize_latent_embeddings(model, checkpoints_dir, step):

    print("Computing TSNE")
    #latent_embedding = model.quantizer.
    spk_embedding = list(spk_embedding.parameters())[0].cpu().detach().numpy()
    spk_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(spk_embedding)

    with open(checkpoints_dir + '/spk_ids') as  f:
       speakers_dict = json.load(f)

    ids2speakers = {v:k for (k,v) in speakers_dict.items()}
    speakers = list(speakers_dict.keys())
    y = spk_embedding[:,0]
    z = spk_embedding[:,1]

    fig, ax = plt.subplots()
    ax.scatter(y, z)

    for i, spk in enumerate(speakers):
        ax.annotate(spk, (y[i], z[i]))

    path = checkpoints_dir + '/step' + str(step) + '_speaker_embedding.png'
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

