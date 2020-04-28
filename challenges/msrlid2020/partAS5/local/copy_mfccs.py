import sys,os
import numpy as np

src_dir = sys.argv[1]
vox_dir = sys.argv[2]

src_dir += '/cleaned'

files = sorted(os.listdir(src_dir))
num_files = len(files)

ctr = 0
for file in files:
 if file[0] =='.':
   continue
 try:
   mfcc = np.loadtxt(   src_dir + '/' + file )
   fname = file.split('.mfcc')[0] + '.feats.npy'
   np.save(vox_dir + '/festival/falcon_mfcc/' + fname, mfcc)
   ctr += 1
   if ctr % 50 == 1:
      print("Processed ", ctr, " files of ", num_files)


 except ValueError:
   print("Problem with this file ", file)
   sys.exit()
