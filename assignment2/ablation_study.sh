#! /bin/sh.

budget=50000
n_repetitions=4
n_cores=4

arc=32_32
lr=0.001
pol=egreedy
eps=0.05
wd=False

bs=5000
up_step=50

# DQN
#python3 deepqn_m.py --experience_replay --target_network --budget $budget --n_repetitions $n_repetitions \
#        --n_cores $n_cores --architecture $arc --learning_rate $lr --target_update_step $up_step --buffer_size $bs \
#        --policy $pol --epsilon $eps --with_decay $wd

# DQN - TN
python3 deepqn_m.py --experience_replay --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores \
        --architecture $arc --learning_rate $lr --buffer_size $bs --policy $pol --epsilon $eps --with_decay $wd

# DQN - EP
#python3 deepqn_m.py --target_network --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores \
        #--architecture $arc --learning_rate $lr --target_update_step $up_step --policy $pol --epsilon $eps \
        #--with_decay $wd

# DQN - TN - EP
#python3 deepqn_m.py --budget $budget --n_repetitions $n_repetitions --n_cores $n_cores --architecture $arc \
        #--learning_rate $lr --policy $pol --epsilon $eps --with_decay $wd




