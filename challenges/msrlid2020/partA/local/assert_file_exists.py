import sys, os

fname = sys.argv[1]
feats_dir = sys.argv[2]

f = open(fname)
for line in f:
   line = line.split('\n')[0]
   try:
     assert os.path.exists(feats_dir + '/' + line + '.feats.npy')
   except AssertionError:
     print("This file is missing ", line)

