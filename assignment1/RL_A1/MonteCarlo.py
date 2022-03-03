#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            # TO DO: Add own code
            b = np.random.random_sample()
            if b < epsilon:
                a = np.random.randint(0, self.n_actions)  # Replace this with correct action selection
            else:
                a = argmax(self.Q_sa[s])


        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            # TO DO: Add own code
            probs = softmax(self.Q_sa[s], temp)
            a = np.random.choice(self.n_actions, p=probs)  # Replace this with correct action selection

        return a

    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        Tep = len(actions)
        G = 0
        for i in reversed(range(Tep)):
            G= rewards[i] + self.gamma*G
            self.Q_sa[states[i],actions[i]] += self.learning_rate*(G-self.Q_sa[states[i],actions[i]])
        pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep '''

    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    time_step = 0

    # TO DO: Write your n-step Q-learning algorithm here!
    rewards_all = []
    while time_step < n_timesteps:
        rewards = []
        actions = []
        states = []
        done = False
        s = env.reset()
        states.append(s)
        for t in range(max_episode_length):
            a = pi.select_action(s, policy, epsilon, temp)
            actions.append(a)
            s_next, r, done = env.step(a)
            time_step += 1
            states.append(s_next)
            rewards.append(r)
            rewards_all.append(r)
            if done or time_step == n_timesteps:
                print('Done')
                break
            else:
                s = s_next
        pi.update(states, actions, rewards)

    # TO DO: Write your Monte Carlo RL algorithm here!
    
    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return rewards_all
    
def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))  
            
if __name__ == '__main__':
    test()
