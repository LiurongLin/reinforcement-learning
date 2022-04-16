import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import argparse
from multiprocessing import Pool

from tqdm import tqdm

class Policy_NN:
    def __init__(self, state_shape=(4,), n_actions=2, lr=1e-3, gamma=0.9, arc=[16,64,16], 
                 activation=None):
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.gamma = gamma
        self.lr  = lr
        self.arc = arc
        if activation is None:
            self.activation = ['relu']*len(self.arc)
        else:
            self.activation = activation
        
        # Initialize neural network
        self.build_NN()
        
        # Use Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        # Use MSE loss
        self.loss = tf.keras.losses.MSE
        
    def build_NN(self):
        
        # Initialize neural network
        self.NN = tf.keras.Sequential()
        
        # Input layer with shape (4,)
        self.NN.add(tf.keras.Input(shape=self.state_shape))
        
        # Densely-connected layers
        for nodes_i, activation_i in zip(self.arc, self.activation):
            self.NN.add(tf.keras.layers.Dense(nodes_i, activation=activation_i))
        
        # Output layer with softmax activation, probability of taking each action
        self.NN.add(tf.keras.layers.Dense(self.n_actions, activation='softmax'))
        
        # Show the architecture of the NN
        #self.NN.summary()
    
    def loss(self, rewards_batch, actions_batch, states_batch):
        
        N_traces  = len(rewards_batch)
        len_trace = len(rewards_batch[0])
        
        grad = 0
        for trace_idx in range(N_traces):
            # Loop over each trace in batch
            states_in_trace  = states_batch[trace_idx]
            actions_in_trace = actions_batch[trace_idx]
            rewards_in_trace = rewards_batch[trace_idx]
            
            # Predict the probabilities for each action
            prob_s = self.NN(states_in_trace)
            # Probabilities for chosen actions
            prob_s_a = pi_s[actions_in_trace]
            
            trace_return = 0
            for obs_idx in range(len_trace)[::-1]:
                # Loop backwards over each observation in trace
                trace_return = rewards_batch[trace_idx][obs_idx] + self.gamma*trace_return
                grad += trace_return * np.log(prob_s_a[obs_idx])
        
        # Multiply by -1 to create a loss function
        grad = - 1/N_traces * grad
        
        return grad
                
                        

class Agent:
    def __init__(self, Policy_NN, budget=50000):
        
        # Number of timesteps to train
        self.budget = budget

        # Add DQN to this class
        self.Policy_NN = Policy_NN

