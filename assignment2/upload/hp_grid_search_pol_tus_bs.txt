#! /bin/sh.

budget=10000
n_repetitions=8
n_cores=4

arc=32
lr=0.01

results_dir=./hp_pol_tus_bs_results
figure_type=pol_tus_bs

for bs in 50 200 800
do
  for up_step in 50 200 800
  do
    for eps in 0.2 0.1 0.05
    do
      python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
              --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
              --buffer_size $bs  --policy egreedy --epsilon $eps --results_dir $results_dir
    done
    python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy egreedy --with_decay True --results_dir $results_dir

    for t in 10 1 0.1
    do
      python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
              --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
              --buffer_size $bs --policy softmax --temp $t --results_dir $results_dir
    done
    python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
            --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step \
            --buffer_size $bs --policy softmax --with_decay True --results_dir $results_dir
  done
done

python3 plot_results.py --results_dir $results_dir --figure_type $figure_type
