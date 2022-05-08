##Reinforcement learning 2021-2022
### Assignment 3: Policy based RL

This README contains instructions on how to produce the results presented in our report.

----------------------------------------------------------------------------------------
To produce the results of exploring plain reinforce with learning rate of [0.01, 0.001, 0.0001] and batch size of [1, 10, 30 ,50],

run `python3 experiment1.py`.


To produce the results of actor-critic with different combination of bootstrapping, baseline subtraction and entropy regulization,

run `python3 experiment2.py`. We have run this experiment with learning rates 0.001 and 0.01.


To produce the results of hyperparameters search on n_boot = 1,

run `python3 experiment_n_boot.py`.

------------------------------------------------------------------------------------------


The rewards are stored as the npy files in a separate folder for each configuration. These are:

`results/experiment1`,

`results/experiment2` and

`results/experiment_n_boot=1`.

The learning curve plots for each experiment are automatically generated and stored in their corresponding folder.

The gradients variances for the neural network parameters are stored as the npy files in the `results/experiment2/grad_vars` folder.

The histograms of the variances are automatically generated and also stored in `results/experiment2/grad_vars`.



Note: the expected runtime of each experiment is 6-10 hours.
