import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment


### Text Processing Stuff
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

def populate_stressarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary") 
       sys.exit()
    #print(feats_dict)
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[stress] for stress in line]
    feats = np.array(feats)
    return feats

def populate_qF0sarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
       print("Expected a feature dictionary") 
       sys.exit()
    #print(feats_dict)
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[qF0] for qF0 in line]
    feats = np.array(feats)
    return feats

def populate_phonesNspkarray(fname, feats_dir, feats_dict, spk_dict):
    if feats_dict is None:
       print("Expected a feature dictionary")
       sys.exit()

    filename = os.path.basename(fname)
    spk = filename.split('_')[0]
    spk_id = spk_dict[spk]
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats, spk_id


def populate_phonesNspkarray(fname, feats_dir, feats_dict, spk_dict):
    if feats_dict is None:
       print("Expected a feature dictionary")
       sys.exit()

    filename = os.path.basename(fname)
    spk = filename.split('_')[0]
    spk_id = spk_dict[spk]
    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats  = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats, spk_id


### Data Source Stuff
class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None, stress_dict=None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

        self.stress_dict = stress_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'phones':
            return populate_phonesarray(fname, self.feats_dir, self.feats_dict)
        if self.feat_name == 'stress':
            return populate_stressarray(fname, self.feats_dir, self.stress_dict)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

class categorical_datasource_spk(categorical_datasource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict, spk_dict):
        super(categorical_datasource_spk, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

        #self.spk_dict = defaultdict(lambda: len(self.spk_dict))
        self.spk_dict = spk_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'phonesNspk':
            self.feats_dir = 'vox/festival/falcon_phones'
            fname = self.feats_dir + '/' + os.path.basename(fname)
            return populate_phonesNspkarray(fname, self.feats_dir, self.feats_dict, self.spk_dict)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

    def get_spkdict(self):
        return self.spk_dict
 
class categorical_datasource_qF0(categorical_datasource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict, qF0_dict):
        super(categorical_datasource_qF0, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

        self.qF0_dict = qF0_dict

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'
        if self.feat_name == 'qF0':
            return populate_qF0sarray(fname, self.feats_dir, self.qF0_dict)
        else:
            print("Unknown feature type: ", self.feat_name)
            sys.exit()

    def get_spkdict(self):
        return self.spk_dict

class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)


#### Visualization Stuff

def visualize_phone_embeddings(model, checkpoints_dir, step):
    return
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


def visualize_speaker_embeddings(model, checkpoints_dir, step):
    return
    print("Computing TSNE")
    #latent_embedding = model.quantizer.
    spk_embedding = list(model.spk_embedding.parameters())[0].cpu().detach().numpy()
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

    x_array = [x for (x,_,_) in batch]
    chars = [x for (x,_) in x_array]
    mels =  [x for (_, x,_) in batch]
    linears = [x for (_,_,x) in batch]
    spk_ids =  [x for (_,x) in x_array]

    input_lengths = [len(c) for c in chars]
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
      a = np.array([_pad(c, max_input_len) for c in chars], dtype=np.int)
    except Exception as e:
      print("Exception here : ", chars)
      print(e)
      sys.exit()
    x_batch = torch.LongTensor(a)
    spk_batch = torch.LongTensor(spk_ids)

    input_lengths = torch.LongTensor(input_lengths)
    b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(linear, max_target_len) for linear in linears],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    return x_batch, spk_batch, input_lengths, mel_batch, y_batch


# We get (x_array, spk_id), stress, y, mel
def collate_fn_spkNstress(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    x_array = [x for (x,_,_,_) in batch]
    chars = [x for (x,_) in x_array]
    mels =  [x for (_, _, x,_) in batch]
    linears = [x for (_,_,_,x) in batch]
    spk_ids =  [x for (_,x) in x_array]
    stress = [x for (_,x,_,_) in batch]

    input_lengths = [len(c) for c in chars]
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
      a = np.array([_pad(c, max_input_len) for c in chars], dtype=np.int)
      a_s = np.array([_pad(c, max_input_len) for c in stress], dtype=np.int)

    except Exception as e:
      print("Exception here : ", chars)
      print(e)
      sys.exit()

    x_batch = torch.LongTensor(a)
    spk_batch = torch.LongTensor(spk_ids)
    stress_batch = torch.LongTensor(a_s)

    input_lengths = torch.LongTensor(input_lengths)
    b = np.array([_pad_2d(mel, max_target_len) for mel in mels],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(linear, max_target_len) for linear in linears],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)


    return x_batch, spk_batch, stress_batch, input_lengths, mel_batch, y_batch




class LocalControlDataset(object):
    def __init__(self, X, c, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y
        self.c = c

    def __getitem__(self, idx):
        return self.X[idx], self.c[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

