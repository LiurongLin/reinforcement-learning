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
    ax.legend(ncol=1, fontsize=15)
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
    for i in save_array:
        # A value of 0 indicates the end of a repetition
        if i == 0:
            rep_rewards.append(rep)
            rep = np.empty(0)
        else:
            # Turn the episode reward into a range and append to the repetition array
            rep = np.append(rep, np.arange(1, i + 1), axis=0)
    return np.array(rep_rewards)


def select_runs(save_dir, **kwargs):
    """
    Select all runs in a save dir that satisfy the conditions given by the kwargs.
    """
    all_run_paths = glob.glob(os.path.join(save_dir, '*'))
    selected_paths = []
    for run_path in all_run_paths:
        select = True
        for key in kwargs:
            condition = f'{key}={kwargs[key]}'
            if condition not in run_path:
                select = False
        if select:
            selected_paths.append(run_path)
    arrays = []
    for run_path in selected_paths:
        save_array = np.load(run_path)
        arrays.append(saved_array_to_plot_array(save_array))
    return arrays


if __name__ == '__main__':
    results_dir = './results/experiment1'
    reward_files = glob.glob(os.path.join(results_dir, '*'))

    linetypes = ['C0-', 'C1-', 'C2-', 'C3-']

    for lr in [1e-2, 1e-3, 1e-4]:
        rewards, labels = [], []
        for batch_size in [1, 10, 30, 50]:
            selected_runs = select_runs(results_dir, lr=lr, batch_size=batch_size)
            title = f'REINFORCE reward progression, lr = {lr}'
            save_file = f'lr={lr}.png'

            # Average rewards over all iterations
            mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
            rewards.append(mean_rewards)

            # Store label for each line
            labels.append(f'batch_size={batch_size}')

        plot_rewards(rewards, config_labels=labels, save_file=f'{results_dir}/{save_file}',
                     linetypes=linetypes, title=title)
