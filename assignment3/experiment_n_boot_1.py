from multiprocessing import Pool
from REINFORCE import pool_function, save_results, get_numpy_file

def adjust_hp(**kwargs):
    
    # Standard hyper-parameters
    args_dict = {'lr': 1e-3,
                 'actor_arc': (64, 64),
                 'critic_arc': (64, 64),
                 'n_boot': 1,
                 'with_bootstrap': True,
                 'with_baseline': False,
                 'with_entropy': False,
                 'eta': 0.01,
                 'budget': 500000,
                 'batch_size': 10,
                 'n_repetitions': 4,
                 'n_cores': 4,
                 'results_dir': './results/experiment_n_boot=1'}

    for key in kwargs:
        args_dict[key] = kwargs[key]

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

# Change one hyper-parameter
adjust_hp(lr=1e-4)
adjust_hp(batch_size=50)
adjust_hp(critic_arc=(64,64,64))
adjust_hp(critic_arc=(64,))
adjust_hp(with_entropy=True, eta=0.1)

# Plot the results
results_dir_standard = './results/experiment2'
results_dir = './results/experiment_n_boot=1'
plot_results_exp_n_boot_1(results_dir_standard, results_dir)