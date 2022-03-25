#! /bin/sh.



for up_step in 10 100 1000
do
	for lr in 0.01 0.005 0.001
	do
		for bs in 10 100 1000
		do
			for arc in 32 64 32_32
			do
				for pol in 'egreedy' 'softmax'
				do
					if (($pol == 'egreedy'))
					then
						for eps in 0.2 0.1 0.05
						do
							python3 deepqn_m.py --experience_replay --target_network --target_update_step $up_step  --learning_rate $lr --buffer_size $bs --architecture $arc --policy 							$pol --epsilon $eps
						done
					
					else		
						for temp in 10 1 0.1
						do
							python3 deepqn_m.py --experience_replay --target_network --target_update_step $up_step  --learning_rate $lr --buffer_size $bs --architecture $arc --policy 							$pol --temp $temp
						done
					fi		
				done
			done
		done
	done
done 




