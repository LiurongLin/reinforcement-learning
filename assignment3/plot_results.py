import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.signal import savgol_filter

plt.rcParams.update({'font.size': 17})


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed
    window: size of the smoothing window '''
    return savgol_filter(y, window, poly)


def plot_rewards(rewards, config_labels, save_file=None, title='DQN mean reward progression', linetypes=None,
                 ylim=(0, 200), show=False):
    if linetypes == None:
        linetypes = ['-'] * len(rewards)

    budget = rewards[0].shape[0]
    steps = np.arange(budget)
    smoothing_window = budget // 10 + 1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        ax.plot(steps, smooth(rewards[i], smoothing_window), linetypes[i], label=config_labels[i])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean reward')
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=1, fontsize=15)
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


def select_runs(save_dir, grad_vars=False, **kwargs):
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
        print('path', run_path)
        save_array = np.load(run_path)
        if grad_vars:
            arrays.append(save_array.reshape((8, -1))[:, :-1])
        else:
            arrays.append(saved_array_to_plot_array(save_array))
    return arrays


def plot_results_exp1(results_dir):
    linetypes = ['C0-', 'C0--', 'C0-.', 'C0:',
                 'C1-', 'C1--', 'C1-.', 'C1:',
                 'C2-', 'C2--', 'C2-.', 'C2:']

    rewards, labels = [], []

    for lr in [1e-2, 1e-3, 1e-4]:
        for batch_size in [1, 10, 30, 50]:
            selected_runs = select_runs(results_dir, lr=lr, batch_size=batch_size)

            # Average rewards over all iterations
            mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
            rewards.append(mean_rewards)

            # Store label for each line
            labels.append(f'lr={lr}, bs={batch_size}')

    title = f'REINFORCE reward progression'
    save_file = f'experiment1.png'

    plot_rewards(rewards, config_labels=labels, save_file=f'{results_dir}/{save_file}',
                 linetypes=linetypes, title=title)


def plot_results_exp2(results_dir):
    linetypes = ['C0-', 'C0--', 'C0-.', 'C0:',
                 'C1-', 'C1--', 'C1-.', 'C1:']

    for wba in [True, False]:
        mean_rewards_per_config, reward_labels = [], []
        for wen in [True, False]:
            mean_grad_vars_per_config, grad_var_labels = [], []
            for n in [0, 1, 10, 50]:
                wbo = True if n != 0 else False
                rewards = select_runs(results_dir, n_boot=n, with_baseline=wba, with_bootstrap=wbo,
                                      with_entropy=wen)

                # Average rewards over all iterations
                mean_rewards = np.mean(np.concatenate(rewards), axis=0)
                mean_rewards_per_config.append(mean_rewards)

                grad_vars = select_runs(os.path.join(results_dir, 'grad_vars'), grad_vars=True, n_boot=n,
                                        with_baseline=wba, with_bootstrap=wbo, with_entropy=wen)

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
                suffix = ' with entropy'
            elif wen and not wba:
                suffix = ' with baseline'
            else:
                suffix = ''
            title = 'Gradient variance of REINFORCE' + suffix
            save_file = f'experiment2_wba={wba}_wen={wen}.png'

            plot_grad_vars(mean_grad_vars_per_config, config_labels=grad_var_labels,
                           save_file=os.path.join(results_dir, 'grad_vars', save_file),
                           title=title)

        wba_string = ' with baseline' if wba else ' without baseline'
        title = 'Reward progression of REINFORCE' + wba_string
        save_file = f'experiment2_wba={wba}.png'

        plot_rewards(mean_rewards_per_config, config_labels=reward_labels,
                     save_file=os.path.join(results_dir, save_file), linetypes=linetypes, title=title)


def plot_grad_vars(grad_vars, config_labels, save_file=None, title='DQN mean reward progression', linetypes=None,
                   show=False):
    if linetypes == None:
        linetypes = ['-'] * len(grad_vars)

    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        var = np.array(grad_vars[i])
        var = np.log10(var[var > 0])
        ax.hist(var, bins=40, density=True, histtype='step', linestyle=linetypes[i], label=config_labels[i])
    ax.set_xlabel('log(gradient)')
    ax.set_ylabel('density')
    ax.set_title(title)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, 2.0)
    ax.legend(loc='upper left', ncol=1, fontsize=15)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
    if show:
        plt.show()
    plt.close()



if __name__ == '__main__':
    # results_dir = './results/experiment1'
    # plot_results_exp1(results_dir)

    results_dir = './results/experiment2'
    plot_results_exp2(results_dir)
