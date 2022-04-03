import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

from Helper import smooth

plt.rcParams.update({'font.size': 17})


def plot_rewards(rewards, config_labels, save_file=None, title='DQN mean reward progression', linetypes=None, ylim=(0,170)):
    if linetypes == None:
        linetypes = ['-']*len(rewards)
    
    budget = rewards[0].shape[0]
    steps = np.arange(budget)
    smoothing_window = budget // 10 + 1
    n_configs = len(config_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        ax.plot(steps, smooth(rewards[i], smoothing_window), linetypes[i], label=config_labels[i])
        # ax.plot(steps, rewards[i], label=config_labels[i])
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean reward')
    ax.set_ylim(ylim)
    ax.set_title(title)
    #ax.legend(ncol=3, fontsize=15)
    ax.legend(ncol=1, fontsize=15)
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=300)
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

def plot_pol_tus_bs_rewards(results_dir):
    """
    Plot the learning curves for each exploration strategy as separate figures.
    In each figure, a line is drawn for each combination of buffer size and target network update step.
    """
    
    linetypes = ['C0-', 'C1-', 'C2-', 
                 'C0--', 'C1--', 'C2--', 
                 'C0-.', 'C1-.', 'C2-.']
    
    # Run for each epsilon exploration parameter
    for eps in [0.2, 0.1, 0.05, 'wd']:
        
        # Run for each combination of buffer size and target update step
        rewards, labels = [], []
        for bs in [50,200,800]:
            for tus in [50,200,800]:
                
                # Select a specific run
                if eps == 'wd':
                    # Linear annealing
                    selected_runs = select_runs(results_dir, pol='egreedy', bs=bs, tus=tus, wd=True)
                    title = r'DQN reward progression - $\epsilon$-greedy policy (annealing $\epsilon$)'
                    save_file = f'egreedy_wd.png'
                else:
                    selected_runs = select_runs(results_dir, pol='egreedy', eps=eps, bs=bs, tus=tus, wd=False)
                    title = r'DQN reward progression - $\epsilon$-greedy policy ($\epsilon=$'+str(eps)+')'
                    save_file = f'egreedy_eps={eps}.png'
                
                # Average rewards over all iterations
                mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
                rewards.append(mean_rewards)
                
                # Store label for each line
                labels.append(f'bs={bs}, tus={tus}')
        
        plot_rewards(rewards, config_labels=labels, save_file=f'{results_dir}/{save_file}', 
                     linetypes=linetypes, title=title)

    # Run for each temperature exploration parameter
    for t in [10.0, 1.0, 0.1, 'wd']:
        
        # Run for each combination of buffer size and target update step
        rewards, labels = [], []
        for bs in [50,200,800]:
            for tus in [50,200,800]:
                
                # Select a specific run
                if t == 'wd':
                    # Linear annealing
                    selected_runs = select_runs(results_dir, pol='softmax', bs=bs, tus=tus, wd=True)
                    title = r'DQN reward progression - softmax policy (annealing $\tau$)'
                    save_file = f'softmax_wd.png'
                else:
                    selected_runs = select_runs(results_dir, pol='softmax', t=t, bs=bs, tus=tus, wd=False)
                    title = r'DQN reward progression - softmax policy ($\tau=$'+str(t)+')'
                    save_file = f'softmax_t={t}.png'
                
                # Average rewards over all iterations
                mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
                rewards.append(mean_rewards)
                
                # Store label for each line
                labels.append(f'bs={bs}, tus={tus}')
        
        plot_rewards(rewards, config_labels=labels, save_file=f'{results_dir}/{save_file}', 
                     linetypes=linetypes, title=title)

def plot_arc_lr_rewards(results_dir):

    linetypes = ['C0-', 'C1-', 'C2-', 
                 'C0--', 'C1--', 'C2--', 
                 'C0-.', 'C1-.', 'C2-.']
    
    rewards, labels = [], []
    
    # Run for each architecture
    for arc in ['32', '64', '32_32']:
        
        # Run for each learning rate
        for lr in [0.001, 0.005, 0.01]:
            
            # Select a specific run
            selected_runs = select_runs(results_dir, lr=lr, arc=arc)

            # Average rewards over all iterations
            mean_rewards = np.mean(np.concatenate(selected_runs), axis=0)
            rewards.append(mean_rewards)

            # Store label for each line
            labels.append(f'lr={lr}, arc={arc}')

    title = r'DQN reward progression - grid-search lr + arc'
    save_file = f'{results_dir}/grid_search_lr_arc.png'
    plot_rewards(rewards, config_labels=labels, save_file=save_file, 
                 linetypes=linetypes, title=title)
    

def plot_ablation_rewards(results_dir):
    
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
        
        if ('we=True' in all_run_paths[idx]) and ('wtn=True' in all_run_paths[idx]):
            label = 'DQN'
        elif ('we=True' in all_run_paths[idx]) and ('wtn=False' in all_run_paths[idx]):
            label = 'DQN-TN'
        elif ('we=False' in all_run_paths[idx]) and ('wtn=True' in all_run_paths[idx]):
            label = 'DQN-ER'
        elif ('we=False' in all_run_paths[idx]) and ('wtn=False' in all_run_paths[idx]):
            label = 'DQN-TN-ER'
        labels.append(label)

    plot_rewards(mean_rewards, config_labels=labels, save_file=f'{results_dir}/dqn_rewards_ablation.png')

        
def read_arguments():
    parser = argparse.ArgumentParser()

    # All arguments to expect
    parser.add_argument('--results_dir', nargs='?', type=str, default='./results', help='Directory where results are saved')
    parser.add_argument('--figure_type', nargs='?', type=str, default='pol_tus_bs', help='Type of figure to plot')

    # Read the arguments in the command line
    args = parser.parse_args()

    args_dict = vars(args)  # Create a dictionary

    return args_dict
    
if __name__ == '__main__':

    # Read the arguments in the command line
    args_dict = read_arguments()
    
    if args_dict['figure_type'] == 'pol_tus_bs':
        plot_pol_tus_bs_rewards(args_dict['results_dir'])
    
    elif args_dict['figure_type'] == 'arc_lr':
        plot_arc_lr_rewards(args_dict['results_dir'])
        
    elif args_dict['figure_type'] == 'ablation':
        plot_ablation_rewards(args_dict['results_dir'])
