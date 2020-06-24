data_root=/home2/srallaba/data/wilderness
lang=ACCIBS

VOXDIR=vox

mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu ${lang} ${lang} || exit 1

# Take up to 2000 of the best 85%
if [ -f ${data_root}/${lang}/aligned/etc/txt.done.data.rfs ]
  then
    cp -pr ${data_root}/${lang}/aligned/etc/txt.done.data.rfs etc/txt.done.data
  else
    awk '{if ($(NF-1) !~ /nan/) print $(NF-1),$0}' ${data_root}/${lang}/aligned/etc/txt.done.data |
    sort -n |
    head -2000 |
    sed 's/^[^(]*(/(/' | sort > etc/txt.done.data

    awk '{print $2}' etc/txt.done.data |
    while read x
     do
       ln -s ${data_root}/${lang}/aligned/wav/$x.wav wav
     done
  $FESTVOXDIR/src/grapheme/make_cg_grapheme 

fi

./bin/do_build build_prompts || exit 1
./bin/do_build get_phseq || exit 1

cat etc/txt.done.data | awk '{print $2}' | sed 's/[ \t]*$//' > fnames
./bin/traintest fnames
./bin/traintest ehmm/etc/txt.phseq.data
cp fnames.test fnames.val

python3.5 $FALCONDIR/dataprep_addmspec.py fnames .
python3.5 $FALCONDIR/dataprep_addlspec.py fnames .
python3.5 $FALCONDIR/dataprep_addphones.py ehmm/etc/txt.phseq.data .

python3.5 local/train_phones.py --exp-dir exp/exp_baseline_${lang} --gpu-id 2 --conf conf/tacotron.conf > log_phones

python3.5 local/synthesize_phones.py exp/exp_baseline_ACCIBS/checkpoints/checkpoint_step0150000.pth vox/ehmm/etc/txt.phseq.data.test.head.1 exp/exp_baseline_ACCIBS/samples_tacotron

echo "Done"
