from multiprocessing import Pool
from REINFORCE import pool_function, save_results, get_numpy_file
import os

for wba in [True, False]:
    for wen in [True, False]:
        for n in [0, 1, 10, 50]:
            wbo = True if n != 0 else False
            args_dict = {'lr': 1e-3,
                         'actor_arc': (64, 64),
                         'critic_arc': (64, 64),
                         'n_boot': n,
                         'with_bootstrap': wbo,
                         'with_baseline': wba,
                         'with_entropy': wen,
                         'eta': 0.01,
                         'budget': 500000,
                         'batch_size': 10,
                         'n_repetitions': 8,
                         'n_cores': 8,
                         'results_dir': './results/experiment2'}

            # args_dict is placed in a list to avoid unpacking the dictionary
            params = args_dict['n_repetitions'] * [[args_dict]]

            # Run the repetitions on multiple cores
            pool = Pool(args_dict['n_cores'])
            results_per_rep = pool.starmap(pool_function, params)
            pool.close()
            pool.join()

            rewards_per_rep = [results_per_rep[i][0] for i in range(len(results_per_rep))]
            grad_vars_per_rep = [results_per_rep[i][1] for i in range(len(results_per_rep))]

            # Create filename and directory to save the rewards
            rewards_filename = get_numpy_file(args_dict['results_dir'], args_dict)

            # Save the rewards to a .npy file
            save_results(rewards_per_rep, rewards_filename)

            grad_vars_filename = get_numpy_file(os.path.join(args_dict['results_dir'], 'grad_vars'), args_dict)
            save_results(grad_vars_per_rep, grad_vars_filename)
