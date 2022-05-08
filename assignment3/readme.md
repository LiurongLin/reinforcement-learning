##Reinforcement learning 2021-2022
### Assignment 3: Policy based RL

This README contains instructions on how to produce the results presented in our report.

To produce the results of exploring plain reinforce with learning rate of [0.1, 0.01, 0.001, 0.0001] and batch size of [1, 10, 30 ,50]
run `experiment1.py`

To produce the results of actor-critic with different combination of bootstrapping, baseline subtraction and entropy regulization.
run `experiment2.py`

The rewards are stored as the npy files in results/experiment1(2) folder for each configuration.
The learning curve plots for each experiment are automatically generated and stored in results/experiment1(2) folder

The gradients variances for the nueral network parameters are stored as the npy files in results/experiment2/grad_vars folder
The histograms of the variances are automatically generated and stored in results/experiment2/grad_vars folder


Note: the expected runtime of each script is 6-10 hours.
