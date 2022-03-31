import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Helper import smooth


def plot_rewards(rewards, config_labels, save_file=None):
    budget = rewards[0].shape[0]
    steps = np.arange(budget)
    smoothing_window = budget // 10 + 1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        ax.plot(steps, smooth(rewards[i], smoothing_window), label=config_labels[i])
        # ax.plot(steps, rewards[i], label=config_labels[i])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean reward')
    ax.set_title('DQN mean reward progression')
    ax.legend()
    if save_file is not None:
        plt.savefig(save_file)
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
    # results_dir = './hp_arc_lr_results'
    # results_dir = './hp_pol_tus_bs_results'
    results_dir = './results'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Should contain arrays with shape [budget] which represent the mean of a certain parameter setting
    mean_rewards = []
    labels = []

    all_run_paths = glob.glob(os.path.join(results_dir, '*'))
    arrays = []
    for run_path in all_run_paths:
        save_array = np.load(run_path)
        arrays.append(saved_array_to_plot_array(save_array))
        
    for idx, run in enumerate(arrays):
        mean_rewards.append(np.mean(run, axis=0))
        labels.append(all_run_paths[idx].split('/')[-1].replace('.npy', ''))

    plot_rewards(mean_rewards, config_labels=labels, save_file='dqn_rewards_ablation')
    
    '''
    # 'selected_runs' is a list containing an [n_repetitions, budget] array for every run with the given kwargs

    for eps in [0.2, 0.1, 0.05]:
        selected_runs = select_runs(results_dir, pol='egreedy', eps=eps, wd=False)
        mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
        labels.append(f'egreedy_eps={eps}')

    selected_runs = select_runs(results_dir, pol='egreedy', wd=True)
    mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    labels.append(f'egreedy_wd')

    for t in [10.0, 1.0, 0.1]:
        selected_runs = select_runs(results_dir, pol='softmax', t=t, wd=False)
        mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
        labels.append(f'softmax_t={t}')

    selected_runs = select_runs(results_dir, pol='softmax', wd=True)
    mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    labels.append(f'softmax_wd')

    # for bs in [50, 200, 800]:
    #     selected_runs = select_runs(results_dir, bs=bs)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'replay buffer size={bs}')

    # for tus in [50, 200, 800]:
    #     selected_runs = select_runs(results_dir, tus=tus)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'target update step={tus}')

    # for arc in ['32_lr', '64_lr', '32_32']:
    #     selected_runs = select_runs(results_dir, arc=arc)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'Architecture={arc}')

    # for lr in [0.01, 0.005, 0.001]:
    #     selected_runs = select_runs(results_dir, lr=lr)
    #     mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #     labels.append(f'Learning rate={lr}')

    # for lr in [0.01, 0.005, 0.001]:
    #     for arc in ['32_lr', '64_lr', '32_32']:
    #         selected_runs = select_runs(results_dir, lr=lr, arc=arc)
    #         mean_rewards.append(np.mean(np.concatenate(selected_runs), axis=0))
    #         labels.append(f'Learning rate={lr}, arc={arc}')

    # all_run_paths = glob.glob(os.path.join(results_dir, '*'))
    # arrays = []
    # for run_path in all_run_paths:
    #     save_array = np.load(run_path)
    #     arrays.append(saved_array_to_plot_array(save_array))
    # for idx, run in enumerate(arrays):
    #     mean_rewards.append(np.mean(run, axis=0))
    #     labels.append(idx)
    #     print(idx, all_run_paths[idx])

    # Plotting
    plot_rewards(mean_rewards, config_labels=labels, save_file='dqn_rewards')
    '''