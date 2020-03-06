# Source the kaldi path
mfcc_folder=$1
feats_dir=$2

rm -r ${feats_dir}

mkdir -p ${feats_dir}/cleaned
mkdir -p ${feats_dir}/raw

# Get the mfccs and accomodate in a single file
for file in ${mfcc_folder}/raw_*.ark
do
 fname=$(basename "$file" .ark)
 cat ${mfcc_folder}/${fname}.scp | while read f
 do
   n=`echo "${f}" | cut -d ' ' -f 1`
   echo $f | copy-feats scp:- ark,t:- | apply-cmvn --norm-vars=true --utt2spk=ark:data/train/utt2spk scp:${mfcc_folder}/cmvn_train.scp ark:- ark,t:- | add-deltas ark:- ark,t:- > ${feats_dir}/raw/${n}.mfcc # | apply-cmvn scp:${mfcc_folder}/../cmvn_${lang}.scp ark:- ark,t:- > ../data/${lang}/raw/${n}.mfcc
   #echo $f | copy-feats scp:- ark,t:- | add-deltas ark:- ark,t:-  | apply-cmvn scp:${mfcc_folder}/../data/train/cmvn.scp scp:${mfcc_folder}/../data/test/cmvn.scp  ark:- ark,t:- > ../feats/raw/${n}.mfcc
   cat ${feats_dir}/raw/${n}.mfcc | sed '/\[$/d' | sed 's/]//g' > ${feats_dir}/cleaned/${n}.mfcc
 done
done
