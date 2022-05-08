from multiprocessing import Pool
from REINFORCE import pool_function, save_results, get_numpy_file
from plot_results import plot_results_exp1

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
                     'budget': 500000,
                     'batch_size': batch_size,
                     'n_repetitions': 8,
                     'n_cores': 8,
                     'results_dir': './results/experiment1'}

        # args_dict is placed in a list to avoid unpacking the dictionary
        params = args_dict['n_repetitions'] * [[args_dict]]

        # Run the repetitions on multiple cores
        pool = Pool(args_dict['n_cores'])
        results_per_rep = pool.starmap(pool_function, params)
        pool.close()
        pool.join()

        rewards_per_rep = [results_per_rep[i][0] for i in range(len(results_per_rep))]

        # Create filename and directory to save the rewards
        rewards_filename = get_numpy_file(args_dict['results_dir'], args_dict)

        # Save the rewards to a .npy file
        save_results(rewards_per_rep, rewards_filename)

# Plot the results
results_dir = './results/experiment1'
plot_results_exp1(results_dir)