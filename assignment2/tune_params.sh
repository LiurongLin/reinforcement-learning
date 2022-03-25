#! /bin/sh.

python3 deepqn_m.py --experience_replay --target_network --target_update_step 10
python3 deepqn_m.py --experience_replay --target_network --target_update_step 100
python3 deepqn_m.py --experience_replay --target_network --target_update_step 1000

python3 deepqn_m.py --experience_replay --target_network --policy egreedy
python3 deepqn_m.py --experience_replay --target_network --policy softmax

python3 deepqn_m.py --experience_replay --target_network --learning_rate 0.01
python3 deepqn_m.py --experience_replay --target_network --learning_rate 0.05
python3 deepqn_m.py --experience_replay --target_network --learning_rate 0.001

python3 deepqn_m.py --experience_replay --target_network --epsilon 0.2
python3 deepqn_m.py --experience_replay --target_network --epsilon 0.1
python3 deepqn_m.py --experience_replay --target_network --epsilon 0.05

python3 deepqn_m.py --experience_replay --target_network --temp 10
python3 deepqn_m.py --experience_replay --target_network --temp 1
python3 deepqn_m.py --experience_replay --target_network --temp 0.1
