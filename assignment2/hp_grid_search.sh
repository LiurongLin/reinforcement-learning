#! /bin/sh.

budget=10000
n_repetitions=8
n_cores=4

# arc 32 64 32_32
# lr 0.01 0.005 0.001

arc=32
lr=0.01

for bs in 10 100 1000
do
  for up_step in 10 100 1000
  do
    for eps in 0.2 0.1 0.05
    do
      python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
              --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
              --buffer_size $bs  --policy egreedy --epsilon $eps
    done
    python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy egreedy --with_decay True
    for t in 10 1 0.1
    do
      python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
              --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
              --buffer_size $bs --policy softmax --temp $t
    done
    python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy softmax --with_decay True
  done
done




