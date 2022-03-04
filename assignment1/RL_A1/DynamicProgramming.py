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
from Helper import argmax


class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s '''
        a = argmax(self.Q_sa[s])
        return a
        
    def update(self, s, a, p_sas, r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        self.Q_sa[s, a] = np.sum(p_sas[s, a] * (r_sas[s, a] + self.gamma * np.max(self.Q_sa, axis=1)))
    

def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)    

    max_error = np.inf
    i = 0

    while max_error > threshold:
        max_error = 0
        for s in range(env.n_states):
            for a in range(env.n_actions):
                x = QIagent.Q_sa[s, a]
                QIagent.update(s, a, env.p_sas, env.r_sas)
                max_error = max(max_error, np.abs(x - QIagent.Q_sa[s, a]))
        
        # Plot current Q-value estimates & print max error
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        print("Q-value iteration, iteration {}, max error {}".format(i, max_error))
        i += 1
     
    return QIagent


def experiment():
    np.random.seed(42)
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    # View optimal policy
    done = False
    s = env.reset()
    rs = []
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        rs.append(r)
        env.render(Q_sa=QIagent.Q_sa, plot_optimal_policy=True, step_pause=0.2)
        s = s_next

    start_state_val = np.max(QIagent.Q_sa[3])
    print(f"State value at of the starting state: {start_state_val}")

    mean_steps = (start_state_val - 35) / -1 + 1
    print('Mean number of timesteps under optimal policy', mean_steps)

    expected_reward = start_state_val / mean_steps
    print("Mean reward per timestep under optimal policy: {}".format(expected_reward))

    # State value at s=3: 18.311
    # Mean steps: 17.684 +- 0.039

    # Every step the reward is -1 except for the last step which is 35
    # So on a average, the return is 16.684 * -1 + 1 * 35 = 18.316
    # which is approximately the state value at the start.

    # The state value V is the expectation value of the return

    # The terminal state has a 100% probability to end up return to the terminal state
    # while the reward of going for terminal state to terminal state is 0
    # an equally valid method is to let the probability for every state be 0 when in the terminal state


if __name__ == '__main__':
    experiment()
