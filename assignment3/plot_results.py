import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.signal import savgol_filter
from tqdm import tqdm

plt.rcParams.update({'font.size': 17})


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def plot_rewards(rewards, config_labels, save_file=None, title='DQN mean reward progression', linetypes=None,
                 ylim=(0, 650), show=False, n_legend_cols=2):
    if linetypes == None:
        linetypes = ['-'] * len(rewards)

    budget = rewards[0].shape[0]
    steps = np.arange(budget)
    smoothing_window = budget // 50 + 1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in tqdm(range(n_configs), desc='Smoothing rewards', total=n_configs):
        ax.plot(steps, smooth(rewards[i], smoothing_window), linetypes[i], label=config_labels[i])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean episode return')
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=n_legend_cols, fontsize=15)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
    if show:
        plt.show()
    plt.close()


def saved_array_to_plot_array(save_array):
    """
    Convert a saved result into an array that can be plotted.
    """
    rep_rewards = []
    rep = np.empty(0)
    reps = 0
    for i in save_array:
        # A value of 0 indicates the end of a repetition
        if i == -1:
            rep_rewards.append(rep)
            rep = np.empty(0)
            reps += 1
            print(f'Completed {reps} repetitions')
        else:
            # Turn the episode reward into a range and append to the repetition array
            rep = np.append(rep, np.arange(1, i + 1), axis=0)
    return np.array(rep_rewards)


def saved_array_to_plot_array_L(save_array, n_repetitions):
    """
        Convert a saved result into an array with repeated elements.add
    """
    end = np.where(save_array == -1)[0]
    #end = np.insert(end, 8, )
    rep_rewards = []
    for i in range(n_repetitions):
        if i == 0:
            rewards = save_array[0:end[i]]
        else:
            rewards = save_array[end[i-1]+1:end[i]]
        rewards = rewards.astype("int64")

        rep_rewards.append(np.repeat(rewards, rewards))
        # print(f"Complete {i+1} repetitions")
    return rep_rewards


def select_runs(save_dir, grad_vars=False, n_repetitions=8, **kwargs):
    """
    Select all runs in a save dir that satisfy the conditions given by the kwargs.
    """
    all_run_paths = glob.glob(os.path.join(save_dir, '*.npy'))
    selected_paths = []
    for run_path in all_run_paths:
        select = True
        for key in kwargs:
            condition = f'{key}={kwargs[key]}_'
            if condition not in run_path:
                select = False
        if select:
            selected_paths.append(run_path)
    arrays = []
    for run_path in selected_paths:
        # print('path', run_path)
        save_array = np.load(run_path)
        if grad_vars:
            arrays.append(save_array.reshape((8, -1))[:, :-1])
        else:
            arrays.append(saved_array_to_plot_array_L(save_array, n_repetitions))
    return arrays


def plot_results_exp1(results_dir):
    linetypes = ['C0-', 'C0--', 'C0-.', 'C0:',
                 'C1-', 'C1--', 'C1-.', 'C1:',
                 'C2-', 'C2--', 'C2-.', 'C2:']

    rewards, labels = [], []

    for lr in [1e-2, 1e-3, 1e-4]:
        for batch_size in [1, 10, 30, 50]:
            selected_runs = select_runs(results_dir, lr=lr, batch_size=batch_size, n_repetitions=8)

            # Average rewards over all iterations
            mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
            rewards.append(mean_rewards)

            # Store label for each line
            labels.append(f'lr={lr}, bs={batch_size}')

    title = f'Learning rate and batch size'
    save_file = f'lr_bs.png'

    plot_rewards(rewards, config_labels=labels, save_file=f'{results_dir}/{save_file}',
                 linetypes=linetypes, title=title, n_legend_cols=3)


def plot_results_exp2(results_dir):
    
    #linetypes = ['C0-', 'C0--', 'C0-.', 'C0:',
    #             'C1-', 'C1--', 'C1-.', 'C1:']
    linetypes = ['C0-', 'C1-', 'C2-', 'C3-',
                 'C0--', 'C1--', 'C2--', 'C3--']

    for wba in [True, False]:
        mean_rewards_per_config, reward_labels = [], []
        mean_grad_vars_per_config, grad_var_labels = [], []
        for wen in [True, False]:
            for n in [0, 1, 10, 50]:
                wbo = True if n != 0 else False
                print(f'wba={wba}, wen={wen}, n={n}')
                rewards = select_runs(results_dir, n_boot=n, with_baseline=wba, with_bootstrap=wbo,
                                      with_entropy=wen, n_repetitions=8)

                # Average rewards over all iterations
                mean_rewards = np.mean(np.concatenate(rewards), axis=0)
                mean_rewards_per_config.append(mean_rewards)

                grad_vars = select_runs(os.path.join(results_dir, 'grad_vars'), grad_vars=True, n_boot=n,
                                        with_baseline=wba, with_bootstrap=wbo, with_entropy=wen, n_repetitions=8)

                # Average rewards over all iterations
                mean_grad_vars = np.mean(np.concatenate(grad_vars), axis=0)
                mean_grad_vars_per_config.append(mean_grad_vars[:4610])  # only use the actor grad_vars

                # Store label for each line
                if n != 0:
                    reward_labels.append(f'en={wen}, n={n}')
                    grad_var_labels.append(f'n={n}')
                else:
                    reward_labels.append(f'en={wen}, no bootstrap')
                    grad_var_labels.append(f'no bootstrap')

            if wba and wen:
                suffix = ' with baseline and entropy'
            elif wba and not wen:
                suffix = ' with baseline'
            elif wen and not wba:
                suffix = ' with entropy'
            else:
                suffix = ''
            title = 'Gradient variance of REINFORCE' + suffix
            save_file = f'experiment2_wba={wba}_wen={wen}.png'

        wba_string = ' with baseline' if wba else ' without baseline'
        title = 'Reward progression of REINFORCE' + wba_string
        save_file = f'experiment2_wba={wba}.png'

        plot_grad_vars(mean_grad_vars_per_config, config_labels=reward_labels,
                       linetypes=linetypes, 
                       save_file=os.path.join(results_dir, 'grad_vars', 'grad_vars_'+save_file),
                       title='Gradient variance of REINFORCE' + wba_string)
        
        plot_rewards(mean_rewards_per_config, config_labels=reward_labels,
                     save_file=os.path.join(results_dir, save_file), 
                     linetypes=linetypes, title=title, n_legend_cols=2)


def plot_single_exp2(results_dir, wba, wen, wbo, n):
    rewards = select_runs(results_dir, n_boot=n, with_baseline=wba, with_bootstrap=wbo,
                          with_entropy=wen, n_repetitions=8)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    selected_ind = np.arange(0, 1000000, 100)
    selected_r = np.array(rewards)[0][:,selected_ind]
    print(np.shape(selected_r))
    ax.plot(selected_r)
    ax.set_xlabel("budget")
    ax.set_ylabel("rewards")
    plt.savefig("single_wba_wen_wbo_50.png", dpi = 300)


def plot_grad_vars(grad_vars, config_labels, save_file=None, title='DQN mean reward progression', linetypes=None,
                   show=False):
    if linetypes == None:
        linetypes = ['-'] * len(grad_vars)

    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        var = np.array(grad_vars[i])
        var = np.log10(var[var > 0])
        ax.hist(var, bins=40, density=True, histtype='step', color=linetypes[i][:2], 
                linestyle=linetypes[i][2:], label=config_labels[i])
    ax.set_xlabel('log(gradient)')
    ax.set_ylabel('density')
    ax.set_title(title)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 2.2)
    ax.legend(loc='upper left', ncol=2, fontsize=15)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_results_exp_n_boot_1(results_dir_standard, results_dir):
    rewards = []
    labels = ['en=False, n=1', 'en=False, lr=1e-4', 'en=False, bs=50', 
              'en=False, critic_arc=(64,64,64)', 'en=False, critic_arc=(64,)', 'en=True, tau=0.1']
    
    selected_runs = select_runs(results_dir_standard, n_boot=1, with_entropy=False, 
                                with_baseline=False, n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)
    
    
    results_dir = './results/experiment_n_boot=1'
    selected_runs = select_runs(results_dir, lr=1e-4, n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)
    
    selected_runs = select_runs(results_dir, batch_size=50, n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)
    
    selected_runs = select_runs(results_dir, critic_arc=(64,64,64), n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)
    
    selected_runs = select_runs(results_dir, critic_arc=(64,), n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)

    selected_runs = select_runs(results_dir, eta=0.1, with_entropy=True, n_repetitions=4)
    mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
    rewards.append(mean_rewards)
        
    plot_rewards(rewards, labels, title='Bootstrap n=1 experiments without baseline', 
                 save_file=f'{results_dir}/n_boot=1_experiments.png', n_legend_cols=2)    



if __name__ == '__main__':
    results_dir = './results/experiment1'
    plot_results_exp1(results_dir)

    results_dir = './results/experiment2'
    plot_results_exp2(results_dir)
    # plot_single_exp2(results_dir, True, True, True, 50)
    
    results_dir_standard = './results/experiment2'
    results_dir = './results/experiment_n_boot=1'
    plot_results_exp_n_boot_1(results_dir_standard, results_dir)