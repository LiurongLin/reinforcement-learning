##Reinforcement learning 2021-2022
### Assignment 2: Deep Q Learning

This README contains instructions on how to produce the results presented in our report.

For producing the mean reward progression for different policies, target network update stepsizes and replay buffer sizes, run 

`bash hp_grid_search_pol_tus_bs.txt`.

For producing the mean reward progression for different neural network architectures and learning rates, run

`bash hp_grid_search_arc_lr.txt`.

For producing the results of the ablation study, run

`bash ablation_study.txt`.

Each script will train the model for several different values of hyperparameters or model features. 
Each script will also create its own results directory and save the episode rewards obtained during training the model.
A separate .npy file is used to store the episode rewards of each model configuration. 
Finally, each script makes a plot containing the smoothed average reward per step of several model configurations.

Note: the expected runtime of each script is 10-20 hours.
