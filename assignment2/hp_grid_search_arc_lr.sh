#! /bin/sh.

budget=1000
n_repetitions=1
n_cores=4

bs=200
up_step=50
pol=egreedy
wd=True

results_dir=./hp_arc_lr_results2
figure_type=arc_lr

for arc in 32 64 32_32
do
  for lr in 0.01 0.005 0.001
  do
    python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy egreedy --with_decay True --results_dir $results_dir
  done
done

python3 plot_results.py --results_dir $results_dir --figure_type $figure_type
