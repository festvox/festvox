import os,sys
import pinyin
import pinyin.cedict

input_file = '../data/Hub/train/train_text.txt'

f = open(input_file, encoding='gb18030')
for line in f:
   line = line.split('\n')[0]
   content = ' '.join(k for k in line.split()[1:])
   print(pinyin.get(content))
   print(content)
