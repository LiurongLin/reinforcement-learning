#! /bin/sh.

budget=30000
n_repetitions=16
n_cores=8

bs=200
up_step=50
pol=egreedy
wd=True

for arc in 32 64 32_32
do
  for lr in 0.01 0.005 0.001
  do
    python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy egreedy --with_decay True
  done
done




