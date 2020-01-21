VOXDIR=vox
spk=libritts

source path.sh

mkdir -p ${VOXDIR}
cd ${VOXDIR}
$FESTVOXDIR/src/clustergen/setup_cg cmu us ${spk} || exit 1

data_path=${data_dir}/train-clean-100
rm -r etc/txt.done.data
find ${data_path} -type f -name "*.wav" | while read line
 do
   fname=$(basename "$line" .wav)
   path=$(dirname "$line")
   tfname=$path/$fname.normalized.txt
   text=`cat $tfname | tr '"' ' ' | tr "'" " "` 
   echo '( ' $fname ' " ' $text ' " ) ' >> etc/txt.done.data
   $ESTDIR/bin/ch_wave -F 16000 -c 0 $line -otype riff -scaleN 0.65 -o wav/$fname.wav
 done


./bin/do_build build_prompts || exit 1
./bin/do_build get_phseq || exit 1
cd ..

python3.5 $FALCONDIR/utils/dataprep_addphones.py ${VOXDIR}/ehmm/etc/txt.phseq.data ${VOXDIR}
cat ${VOXDIR}/ehmm/etc/txt.phseq.data | awk '{print $1}' > ${VOXDIR}/fnames
python3.5 $FALCONDIR/utils/dataprep_addlspec.py ${VOXDIR}/fnames ${VOXDIR}
python3.5 $FALCONDIR/utils/dataprep_addmspec.py ${VOXDIR}/fnames ${VOXDIR}

${VOXDIR}/bin/traintest ${VOXDIR}/fnames 
cp ${VOXDIR}/fnames.test ${VOXDIR}/fnames.val

exit

#python3.5 local/train_phones.py --exp-dir exp/taco_one_phseq 
#python3.5 local/synthesize_phones.py exp/taco_one_phseq/checkpoints/checkpoint_step30000.pth  vox/ehmm/etc/txt.phseq.data.test.head exp/taco_one_phseq/tts_phseq
