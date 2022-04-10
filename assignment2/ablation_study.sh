#! /bin/sh.

budget=50000
n_repetitions=32
n_cores=4

arc=64
lr=0.0001
pol=egreedy
eps=0.05
wd=False

bs=800
up_step=50

results_dir=./ablation_results
figure_type=ablation

# DQN
python3 deepqn.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
        --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step --buffer_size $bs \
        --policy $pol --epsilon $eps --with_decay $wd --results_dir $results_dir

# DQN - TN
python3 deepqn.py --experience_replay --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores \
        --architecture $arc --learning_rate $lr --buffer_size $bs --policy $pol --epsilon $eps \
        --with_decay $wd --results_dir $results_dir

# DQN - EP
python3 deepqn.py --target_network --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores \
        --architecture $arc --learning_rate $lr --target_update_step $up_step --policy $pol --epsilon $eps \
        --with_decay $wd --results_dir $results_dir

# DQN - TN - EP
python3 deepqn.py --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores --architecture $arc \
        --learning_rate $lr --policy $pol --epsilon $eps --with_decay $wd --results_dir $results_dir

python3 plot_results.py --results_dir $results_dir --figure_type $figure_type


