for i in 1 2 3 4 5 6 7 8 9 10
  do
    echo "Doing " $i
    python3.5 local/train_phones_gan0a.py --conf /tmp/tacotron.conf --gpu-id 0 --exp-dir exp/exp_gan0atest_${i} > log_test_${i} 
  done
