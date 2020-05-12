import os, sys

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
from hyperparameters import hparams
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment
from sklearn.metrics import *
import random

### Data Source Stuff
class categorical_datasource(CategoricalDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(categorical_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):

        assert self.feat_type == 'categorical'
        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats'

        if self.feat_name == 'valence':
          f = open(fname)
          for line in f:
             valence = line.split('\n')[0]
             return valence

        elif self.feat_name == 'crying':
          f = open(fname)
          for line in f:
             crying = line.split('\n')[0]
             return crying
        

        elif self.feat_name == 'quants':
            return np.load(fname + '.npy')


class float_datasource(FloatDataSource):

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict = None):
        super(float_datasource, self).__init__(fnames_file, desc_file, feat_name, feats_dir, feats_dict)

    def __getitem__(self, idx):

        fname = self.filenames_array[idx]
        fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'

        if self.feat_name == 'soundnet':
           return np.load(fname)

class filenamesDataset(object):
    def __init__(self, fnames):
       self.fnames = fnames
       self.fnames_array = []
       f = open(self.fnames)
       for line in f:
           line = line.split('\n')[0]
           self.fnames_array.append(line)

    def __getitem__(self, idx):
        return self.fnames_array[idx]

    def __len__(self):
        return len(self.fnames_array)


class ValenceDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)

class CryingDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx]

    def __len__(self):
        return len(self.X)

class ValenceNArousalDataset(object):
    def __init__(self, Xa, Xv, Mel):
        self.Xa = Xa
        self.Xv = Xv
        self.Mel = Mel

    def __getitem__(self, idx):
        return self.Xa[idx], self.Xv[idx], self.Mel[idx]

    def __len__(self):
        return len(self.Xa)

class ValenceselfsupervisedMultitaskDataset(object):
    def __init__(self, X, Mel, quants):
        self.X = X
        self.Mel = Mel
        self.quants = quants

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.quants[idx]

    def __len__(self):
        return len(self.X)


class ValenceDataset_CPCLoss(object):
    def __init__(self, X, Mel, fnames):
        self.X = X
        self.Mel = Mel
        self.labels_file = '/home1/srallaba/challenges/compare2020/ComParE2020_Elderly/lab/labels.csv'
        self.column_num = 5
        self.low, self.medium, self.high, self.fname2label_dict = get_label_arrays(self.labels_file, self.column_num)
        self.fnames = fnames
        self.labels = [0,1,2]
        self.fnamearrays = [self.low, self.medium, self.high]

    def __getitem__(self, idx):
        label = self.X[idx]
        mel = self.Mel[idx]
        fname = self.fnames[idx]
        contrastive_label = get_contrastive_label(self.labels, label)
        random_contrastive_fname = random.choice(self.fnamearrays[contrastive_label])
        contrastive_mel = np.load('vox/festival/falcon_mfcc/' + random_contrastive_fname + '.feats.npy')
        return label, mel, contrastive_mel

    def __len__(self):
        return len(self.X)


def get_contrastive_label(label_array, current_label):

   while True:
     random_element = random.choice(label_array)
     if random_element != current_label:
       return random_element


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x

def collate_fn_valence(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch])
    #print(batch)
    a = np.array([x[0] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return x_batch, mel_batch


def collate_fn_crying(batch):
    """Create batch"""

    a = np.array([x[0] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    b = np.array([x[1] for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)
    return x_batch, mel_batch

def collate_fn_valence_seqlen10(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    seq_len = 100
    max_offsets = [x[1].shape[0] - seq_len for x in batch]
    mel_lengths = [x[1].shape[0] for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch])
    #print(batch)
    a = np.array([x[0] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)
        
    mel_batch = torch.FloatTensor([x[1][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])

    return x_batch, mel_batch


def collate_fn_valence_contrastiveloss(batch):
    """Create batch"""
    r = hparams.outputs_per_step

    # Add single zeros frame at least, so plus 1
    max_target_len_positive  = np.max([len(x[1]) for x in batch])
    max_target_len_negative = np.max([len(x[2]) for x in batch])

    a = np.array([x[0] for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    b = np.array([_pad_2d(x[1], max_target_len_positive) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len_negative) for x in batch],
                 dtype=np.float32)
    mel_batch_negative = torch.FloatTensor(c)

    return x_batch, mel_batch, mel_batch_negative

def collate_fn_valenceNquants(batch):
    """Create batch"""

    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch])
    
    a = np.array([x[0] for x in batch], dtype=np.int)
    xv_batch = torch.LongTensor(a)

    input_lengths = [len(x[2]) for x in batch]
    seq_len = 800

    max_offsets = [x[2].shape[0] - seq_len for x in batch]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]

    quants_batch = torch.FloatTensor( [x[2][offsets[i]:offsets[i] + seq_len] for i,x in enumerate(batch) ] )


    seq_len = 100
    max_offsets = [x[1].shape[0] - seq_len for x in batch]
    mel_lengths = [x[1].shape[0] for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    mel_batch = torch.FloatTensor([x[1][mel_offsets[i]:mel_offsets[i] + seq_len] for i, x in enumerate(batch)])

    return xv_batch, mel_batch, quants_batch


def collate_fn_valenceNarousal(batch):
    """Create batch"""
    r = hparams.outputs_per_step


    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[2]) for x in batch])

    a = np.array([x[0] for x in batch], dtype=np.int)
    xa_batch = torch.LongTensor(a)

    a = np.array([x[1] for x in batch], dtype=np.int)
    xv_batch = torch.LongTensor(a)


    b = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    return xa_batch, xv_batch, mel_batch


# Utility to return predictions
def return_classes(logits, dim=-1):
   _, predicted = torch.max(logits,dim)    
   return predicted.view(-1).cpu().numpy()


def get_metrics(predicteds, targets):
   print(confusion_matrix(targets, predicteds))
   print(classification_report(targets, predicteds))
   print("Accuracy is ", accuracy_score(targets, predicteds))
   fpr, tpr, threshold = roc_curve(targets, predicteds, pos_label=1)
   EER = threshold[np.argmin(np.absolute(tpr-fpr))]
   print("EER is ", EER)
   return recall_score(targets,predicteds,average='macro')




def get_label_arrays(fnames, column_num=None):

   try:
       assert column_num is not None
   except AssertionError:
       print("You need to call this function with a column number for me to read from")
       sys.exit()

   low_array = []
   medium_array = []
   high_array = []
   fname2label_dict = {}

   f = open(fnames)
   cnt = 0
   for line in f:
     if cnt == 0:
         cnt += 1
         continue
     print(line)
     line = line.split('\n')[0]
     fname = line.split(',')[0].split('.')[0]
     try:
       feature = int(line.split(',')[column_num])
     except ValueError:
       continue

     if feature == 0:
        low_array.append(fname)
     elif feature == 1:
        medium_array.append(fname)
     elif feature == 2:
        high_array.append(fname)
     else:
        print("Houston, we have got problems. I think the feature is ", feature)
        sys.exit()

     fname2label_dict[fname] = feature

   f.close()

   return low_array, medium_array, high_array, fname2label_dict
