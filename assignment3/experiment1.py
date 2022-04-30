from multiprocessing import Pool
from REINFORCE import pool_function, save_rewards

for lr in [1e-2, 1e-3, 1e-4]:
    for batch_size in [1, 10, 30, 50]:
        args_dict = {'lr': lr,
                     'actor_arc': (64, 64),
                     'critic_arc': (64, 64),
                     'n_boot': 1,
                     'with_bootstrap': False,
                     'with_baseline': False,
                     'with_entropy': False,
                     'eta': 0.01,
                     'budget': 1000000,
                     'batch_size': batch_size,
                     'n_repetitions': 8,
                     'n_cores': 8,
                     'results_dir': './results/experiment1'}

        # args_dict is placed in a list to avoid unpacking the dictionary
        params = args_dict['n_repetitions'] * [[args_dict]]

        # Run the repetitions on multiple cores
        pool = Pool(args_dict['n_cores'])
        rewards_per_rep = pool.starmap(pool_function, params)
        pool.close()
        pool.join()

        # Save the rewards to a .npy file
        save_rewards(rewards_per_rep, args_dict)
