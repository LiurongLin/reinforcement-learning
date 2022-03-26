import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Helper import smooth


def plot_rewards(rewards, config_labels, budget=10000, save_file=None):
    steps = np.arange(budget)
    smoothing_window = budget // 10 + 1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        ax.plot(steps, smooth(rewards[i], smoothing_window), label=config_labels[i])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean reward')
    ax.set_title('DQN mean reward progression')
    ax.legend()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    plt.close()


def saved_array_to_plot_array(save_array, budget=10000):
    """
    Convert a saved result into an array that can be plotted.
    """
    rewards_array = np.empty((0, budget))
    rep = np.empty(0)
    for i in save_array:
        # A value of 0 indicates the end of a repetition
        if i == 0:
            rewards_array = np.append(rewards_array, rep[None, ...], axis=0)
            rep = np.empty(0)
        else:
            # Turn the episode reward into a range and append to the repetition array
            rep = np.append(rep, np.arange(1, i + 1), axis=0)
    return rewards_array


def select_runs(save_dir, budget=10000, **kwargs):
    """
    Select all runs in a save dir that satisfy the conditions given by the kwargs.
    """
    all_run_paths = glob.glob(os.path.join(save_dir, '*'))
    selected_paths = []
    for key in kwargs:
        for run_path in all_run_paths:
            condition = f'{key}={kwargs[key]}'
            if condition in run_path:
                selected_paths.append(run_path)
    arrays = []
    for run_path in selected_paths:
        save_array = np.load(run_path)
        arrays.append(saved_array_to_plot_array(save_array, budget))
    return arrays


if __name__ == '__main__':
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    budget = 10000

    # Should contain arrays with shape [budget] which represent the mean of a certain parameter setting
    mean_rewards = []
    labels = []

    # 'selected_runs' is a list containing [n_repetitions, budget] arrays for every run with the given kwargs

    # for eps in [0.2, 0.1, 0.05]:
    #     selected_runs = select_runs(results_dir, budget=budget, pol='egreedy', eps=eps)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'egreedy_eps={eps}')
    #
    # selected_runs = select_runs(results_dir, budget=budget, pol='egreedy', wd=True)
    # mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    # labels.append(f'egreedy_wd')
    #
    # for t in [10, 1, 0.1]:
    #     selected_runs = select_runs(results_dir, budget=budget, pol='softmax', t=t)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'softmax_t={t}')
    #
    # selected_runs = select_runs(results_dir, budget=budget, pol='softmax', wd=True)
    # mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    # labels.append(f'softmax_wd')

    for bs in [50, 200, 800]:
        selected_runs = select_runs(results_dir, budget=budget, bs=bs)
        mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
        labels.append(f'replay buffer size={bs}')

    # for tus in [50, 200, 800]:
    #     selected_runs = select_runs(results_dir, budget=budget, tus=tus)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'target update step={tus}')

    # Plotting
    plot_rewards(mean_rewards, config_labels=labels, budget=budget, save_file='dqn_rewards')
