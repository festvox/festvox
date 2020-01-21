
# Build barebones voice first
#sh run.sh


# Align, extract durations and F0s
cd vox
./bin/do_build label || exit 1
./bin/do_clustergen generate_statenames || exit 1
./bin/do_clustergen generate_filters || exit 1
./bin/do_clustergen parallel build_utts || exit 1
cd ..

python3.5 local/quantize_f0.py --binsize 15 vox/f0_ascii vox/f0_ascii_quantized
python3.5 local/make_phonesNqF0s.py vox/durations_phones vox/f0_ascii_quantized vox/qF0sNphones/
python3.5 local/populate_tddfromphonesNqF0s.py vox/qF0sNphones vox/etc/txt.done.data.phonesNqF0s
python3.5 local/dataprep_addphonesNqF0s.py vox/etc/txt.done.data.phonesNqF0s vox/
