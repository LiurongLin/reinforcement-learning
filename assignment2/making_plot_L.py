import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from Helper import smooth

bs_l = [50, 200, 800]
eps_l = [0.2, 0.1, 0.05]
t_l = [10, 1, 0.1]
tus_l = [50, 200, 800]
arc_l = ['32_lr', '64_lr', '32_32']
lr_l = [0.01, 0.005, 0.001]

def make_heatmap(data, filename, xlabel, ylabel, xtick, ytick):
    '''
    make the heatmap of two parameters pairs.
    '''
    heatmap_dir = './heatmap_results'

    if not os.path.exists(heatmap_dir):
        os.mkdir(heatmap_dir)

    plt.figure(figsize = (8,8))
    plt.imshow(data, aspect='auto', vmax = 120, vmin = 20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks([0,1,2], map(str, xtick))
    plt.yticks([0,1,2], map(str, ytick))
    plt.title(filename)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel('Mean rewards', fontsize=15, rotation=270)
    return plt.savefig("{}/{}.png".format(heatmap_dir,filename))


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
    #results_dir = './hp_arc_lr_results'
    results_dir = './hp_pol_tus_bs_results'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Should contain arrays with shape [budget] which represent the mean of a certain parameter setting
    mean_rewards = []
    labels = []

    # arc = 32, lr = 0.01, pol = egreedy, wd = False
    eps_tus = np.zeros((3,3))
    for i , eps in enumerate(eps_l):
        for j, tus in enumerate(tus_l):
            selected_runs = select_runs(results_dir, pol='egreedy', eps=eps, tus=tus, wd=False)
            eps_tus[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = "TUS_vs_EPS"
    make_heatmap(eps_tus, filename, 'Target_network_update_step ', 'Epsilon', tus_l, eps_l)

    eps_bs = np.zeros((3, 3))
    for i, eps in enumerate(eps_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='egreedy', eps=eps, bs = bs, wd=False)
            eps_bs[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = "BS_vs_EPS"
    make_heatmap(eps_bs, filename, 'Buffer_size ', 'Epsilon', bs_l, eps_l)

    tus_bs = np.zeros((3, 3))
    for i, tus in enumerate(tus_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='egreedy', tus=tus, bs=bs, wd=False)
            tus_bs[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = 'BS_vs_TUS(egreedy)'
    make_heatmap(tus_bs, filename, 'Buffer_size','Target_netwrok_update_steps', bs_l, tus_l)

    # arc = 32, lr = 0.01, pol = softmax, wd = False
    t_tus = np.zeros((3, 3))
    for i, t in enumerate(t_l):
        for j, tus in enumerate(tus_l):
            selected_runs = select_runs(results_dir, pol='softmax', t = t, tus=tus, wd=False)
            t_tus[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = 'TUS_vs_T'
    make_heatmap(t_tus, filename, 'Target_network_update_step ', 'Temperature', tus_l, t_l)

    t_bs = np.zeros((3, 3))
    for i, t in enumerate(t_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='softmax', t=t, bs=bs, wd=False)
            t_bs[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = 'T_vs_BS'
    make_heatmap(t_bs, filename, 'Buffer_size ', 'Temperature', bs_l, t_l)

    tus_bs_s = np.zeros((3, 3))
    for i, tus in enumerate(tus_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='softmax', tus=tus, bs=bs, wd=False)
            tus_bs_s[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = 'TUS_vs_BS(softmax)'
    make_heatmap(tus_bs_s, filename, 'Buffer_size', 'Target_netwrok_update_steps', bs_l, tus_l)

    # arc = 32, lr = 0.01, wd = True
    tus_bs_wd_e = np.zeros((3, 3))
    for i, tus in enumerate(tus_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='egreedy', tus=tus, bs=bs, wd=True)
            tus_bs_wd_e[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = "TUS_vs_BS_wd_e"
    make_heatmap(tus_bs_wd_e, filename, 'Buffer_size ', 'Target_netwrok_update_steps', bs_l, tus_l)

    tus_bs_wd_s = np.zeros((3, 3))
    for i, tus in enumerate(tus_l):
        for j, bs in enumerate(bs_l):
            selected_runs = select_runs(results_dir, pol='softmax', tus=tus, bs=bs, wd=True)
            tus_bs_wd_s[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    filename = "TUS_vs_BS_wd_s"
    make_heatmap(tus_bs_wd_s, filename, 'Buffer_size ', 'Target_netwrok_update_steps', bs_l, tus_l)

    # # pol = egreedy, wd = True, tus = 50, bs = 200, eps = 0.2
    # arc_lr = np.zeros((3, 3))
    # for i, arc in enumerate(arc_l):
    #     for j, lr in enumerate(lr_l):
    #         selected_runs = select_runs(results_dir, arc = arc, lr = lr)
    #         arc_lr[i][j] = np.mean(np.mean(np.concatenate(selected_runs), axis=0)[-1000:])
    # filename = "ARC_LR"
    # make_heatmap(arc_lr, filename, 'Architecture', 'Learning rate', arc_l, lr_l)