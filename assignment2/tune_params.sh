#! /bin/sh.




for up_step in 10, 100, 1000
do
	for pol in egreedy softmax
	do
		for eps in 0.2 0.1 0.05
		do
			for lr in 0.01 0.05 0.001
			do
				for temp in 10 1 0.1
				do
				python3 deepqn_m.py --experience_replay --target_network --target_update_step $up_step --policy egreedy $pol  --learning_rate $lr --epsilon $eps --temp $temp
				
				done
			done		
		done
	done
done 




