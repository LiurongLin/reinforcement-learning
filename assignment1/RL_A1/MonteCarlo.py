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

            if np.random.uniform() > epsilon:
                a = argmax(self.Q_sa[s])
            else:
                a = np.random.randint(self.n_actions)  # Replace this with correct action selection

        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")

            a = np.random.choice(self.n_actions, p=softmax(self.Q_sa[s], temp))

        return a
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        T_ep = len(actions)
        G = 0
        for t in np.arange(T_ep)[::-1]:
            G = rewards[t] + self.gamma * G
            self.Q_sa[states[t], actions[t]] += self.learning_rate * (G - self.Q_sa[states[t], actions[t]])


def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    step = 0
    while step < n_timesteps:
        s = env.reset()
        episode_states = []
        episode_rewards = []
        episode_actions = []
        episode_states.append(s)

        for _ in range(max_episode_length):
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
            s_next, r, done = env.step(a)

            episode_states.append(s_next)
            episode_rewards.append(r)
            episode_actions.append(a)

            rewards.append(r)
            step += 1
            if done or step == n_timesteps:
                break
            else:
                s = s_next

            if plot:
                env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                           step_pause=0.1)  # Plot the Q-value estimates during Q-learning execution

        pi.update(episode_states, episode_actions, episode_rewards)

    return rewards
    
def test():
    n_timesteps = 1000
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
