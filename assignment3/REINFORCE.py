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
        

class Agent:
    def __init__(self, Policy_NN, budget=50000):
        
        # Number of timesteps to train
        self.budget = budget

        # Add DQN to this class
        self.Policy_NN = Policy_NN

