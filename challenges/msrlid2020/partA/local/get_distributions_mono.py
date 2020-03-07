import os, sys

# Lets get the distributions for Telugu first

data_dir = '/home1/srallaba/challenges/msrcodeswitch2020/data/'

def print_distribution(labels_file):

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
         print(arr)
         try:
            assert arr[0] == 'T'
         except AssertionError as e:
            print("Something wrong with this honey ", fname)
            raise Exception( e.args )
         mono_array.append(fname)
      elif len(arr) == 2:
         multi_array.append(fname)


   print(len(mono_array), len(multi_array))
   print('\n')


telugu_train_file = data_dir + '/PartA_Telugu/Train/Transcription_LT_Sequence.tsv'
print("Distribution for Telugu Part A Train")
print_distribution(telugu_train_file)

telugu_train_file = data_dir + '/PartA_Telugu/Dev/Transcription_LT_Sequence.tsv'
print("Distribution for Telugu Part A Dev")
print_distribution(telugu_train_file)


