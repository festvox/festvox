import os, sys
import numpy as np


#x // 50

src_dir = sys.argv[1]
tgt_dir = sys.argv[2]
if not os.path.exists(tgt_dir):
   os.mkdir(tgt_dir)

files = sorted(os.listdir(src_dir))
l = len(files)
for i, file in enumerate(files):
   A = np.loadtxt(src_dir + '/' + file, usecols=0)
   A += 10
   A //= 50
   np.savetxt(tgt_dir + '/' + file, A)

   if i % 100 == 0:
      print("Processed ", i, "files out of ", l)
