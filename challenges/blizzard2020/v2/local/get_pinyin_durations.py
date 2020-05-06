import os

'''Formatting
Input format
pau 0 0.050000001 0 
pau 0 0.19 0.050000001 
ey a 0.245 0.19 
m Midsummer 0.32499999 0.245 
ih Midsummer 0.40000001 0.32499999 
d Midsummer 0.465 0.40000001 
s Midsummer 0.59500003 0.465 

Output format
a 0 0.19
Midsummer 0.245 0.80500001
Night 0.80500001 1.225
Dream 1.225 1.6
'''

durations_dir = 'vox/durations_phones/'
duration_files = sorted(os.listdir(durations_dir))
dest_dir = 'vox/durations_pinyin/'
if not os.path.exists(dest_dir):
   os.mkdir(dest_dir)

for file in duration_files:
   print(file)
   word_prev = ''
   start_time = 0
   f = open(durations_dir + '/' + file)
   fname = file.split('.')[0] + '.txt'
   g = open(dest_dir + '/' + fname, 'w')
   for line in f:
      line = line.split('\n')[0]
      word = line.split()[0]
      if word == '0':
         continue
      elif word == word_prev:
         pass
      else:
         end_time = line.split()[2]
         g.write(word + ' ' + str(end_time) + ' ' + str(start_time) + '\n')
         start_time = end_time
         word_prev = word  
   g.close()
   f.close()
