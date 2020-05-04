import os

word2phone_file = '../voices/v0_words/ehmm/etc/txt.phseq.data'
word2tone_file = 'tdd.underscores'
phonesNtones_file = 'tdd.phonesNtones'

# Read word to phone mapping from word2phone_file and put in a dict
f = open(word2phone_file)
word2phones = {}
for line in f:
  line = line.split('\n')[0].split()
  word = line[0].split('_')[1]
  phones = ' '.join(k for k in line[1:])
  word2phones[word] = phones

print(word2phones)

# Read word to tone mapping and replace word by phones
f = open(word2tone_file)
g = open(phonesNtones_file, 'w')
for line in f:
  line = line.split('\n')[0].split()
  print(line)
  fname = line[0]
  wordsNtones = line[1:]
  phonesNtones = ''
  for w_t in wordsNtones:
     if w_t == ',_0':
         phonesNtones += ' ' + w_t
         continue
     print ( "w_t is  ",  w_t)
     w,t = w_t.split('_')[0], w_t.split('_')[1]
     phones = word2phones[w]
     pNt = ' '.join(k + '_' + t for k in phones.split())
     phonesNtones += ' ' + pNt
  print(phonesNtones)
  g.write(fname + ' ' + phonesNtones + '\n')
g.close()

# Write phonesNtones file
