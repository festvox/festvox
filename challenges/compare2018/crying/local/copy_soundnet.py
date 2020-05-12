import sys,os
import numpy as np

src_dir = sys.argv[1]
vox_dir = sys.argv[2]


files = sorted(os.listdir(src_dir))
num_files = len(files)

ctr = 0
for file in files:
 if file[0] =='.':
   continue
 try:
   A = np.load(   src_dir + '/' + file, allow_pickle=True, encoding="latin1" )
   fname = file.split('.')[0] + '.feats.npy'
   a = A['arr_0']
   inp = np.mean(a[4],axis=0)
   np.save(vox_dir + '/festival/falcon_soundnet/' + fname, inp)
   ctr += 1
   if ctr % 50 == 1:
      print("Processed ", ctr, " files of ", num_files)


 except ValueError as e:
   print("Problem with this file ", file, e)
   sys.exit()


print(a.shape)
