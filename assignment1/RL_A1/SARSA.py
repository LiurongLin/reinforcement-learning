#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

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
        
    def update(self,s,a,r,s_next,a_next,done):
        if done:
            G = r
        else:
            G = r + self.gamma * self.Q_sa[s_next, a_next]
        self.Q_sa[s, a] += self.learning_rate * (G - self.Q_sa[s, a])
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    t = 0
    s = env.reset()
    a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

    while t < n_timesteps:
        s_next, r, done = env.step(a)
        a_next = pi.select_action(s_next, policy=policy, epsilon=epsilon, temp=temp)
        pi.update(s, a, r, s_next, a_next, done)
        rewards.append(r)
        t += 1
        if done:
            s = env.reset()
            a = pi.select_action(s, policy=policy, epsilon=epsilon, temp=temp)
        else:
            s = s_next
            a = a_next

        if plot:
            env.render(Q_sa=pi.Q_sa, plot_optimal_policy=True,
                       step_pause=0.1)  # Plot the Q-value estimates during Q-learning execution

    return rewards


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))        
    
if __name__ == '__main__':
    test()
