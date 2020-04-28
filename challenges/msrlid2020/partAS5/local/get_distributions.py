import os, sys
import numpy as np

# Lets get the distributions for Telugu first

data_dir = '/home/srallaba_lid/challenges/msrlid2020/data/'
vox_dir = 'vox'

ftr = vox_dir + '/fnames.train'
fte = vox_dir + '/fnames.dev'

def print_distribution(labels_file, fnames_file):

   ft = open(fnames_file, 'w')
   mono_array = []
   multi_array = []

   f = open(labels_file)
   for line in f:
      line = line.split('\n')[0].split('\t')
      fname = line[0]
      labels = ' '.join(k for k in line[1:])
      arr = []
      for l in labels:
        if l == ' ':
           continue
        arr.append(l)
      arr = list(set(arr))
      if len(arr) == 1:
         mono_array.append(fname)
         lid = 0 #'mono'
      elif len(arr) == 2:
         multi_array.append(fname)
         lid = 1 #'multi'

      fn = vox_dir + '/festival/falcon_lid/' + fname + '.feats'
      np.save(fn, lid)
      ft.write(fname + '\n')

   print(len(mono_array), len(multi_array))
   print('\n')


telugu_train_file = data_dir + '/PartA_Telugu/Train/Transcription_LT_Sequence.tsv'
print("Distribution for Telugu Part A Train")
print_distribution(telugu_train_file, ftr)

telugu_train_file = data_dir + '/PartA_Telugu/Dev/Transcription_LT_Sequence.tsv'
print("Distribution for Telugu Part A Dev")
print_distribution(telugu_train_file, fte)
