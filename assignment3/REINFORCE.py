import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import argparse
from multiprocessing import Pool

from tqdm import tqdm

class Policy:
    def __init__(self, state_shape=(4,), n_actions=2, lr=1e-3, gamma=0.9, arc=[64,64], 
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
        self.NN.summary()
    
    def select_action(self, s):
        
        # Probability of taking each action
        prob_s = np.array(self.NN(s)).flatten()
        
        # Add some noise for exploration
        #prob_s = ...
        
        # Sample action
        a = np.random.choice(self.n_actions, p=prob_s)
        return a
    
    def loss_function(self, s_batch, a_batch, r_batch):
        
        N_traces = len(r_batch)
       
        loss = tf.constant(0, dtype=tf.float32)
        for trace_idx in range(N_traces):
            # Loop over each trace in batch
            s_in_trace = tf.convert_to_tensor(s_batch[trace_idx])
            a_in_trace = a_batch[trace_idx]
            r_in_trace = r_batch[trace_idx]

            len_trace = len(r_in_trace)
            
            # Predict the probabilities for each action
            prob_s = self.NN(s_in_trace)
            # Probabilities for chosen actions
            prob_sa = tf.reduce_sum(tf.one_hot(a_in_trace, self.n_actions)*prob_s, axis=1)
            
            trace_return = 0
            for obs_idx in range(len_trace)[::-1]:
                # Loop backwards over each observation in trace
                trace_return = r_in_trace[obs_idx] + self.gamma*trace_return
                loss += trace_return * tf.math.log(prob_sa[obs_idx])
        
        # Multiply by -1 to create a loss function
        loss = - 1/N_traces * loss
        return loss
    
    def update(self, s_batch, a_batch, r_batch):
    
        with tf.GradientTape() as tape:
            tape.watch(self.NN.trainable_variables)
            
            # Calculate the loss
            loss_value = self.loss_function(s_batch, a_batch, r_batch)
        
        # Backpropagate the gradient to the network's weights
        gradients = tape.gradient(loss_value, self.NN.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.NN.trainable_variables))


class Agent:
    def __init__(self, Policy, budget=50000, max_len_trace=500, batch_size=50):
        
        # Number of timesteps to train
        self.budget = budget
        
        # Maximum length of the trace
        self.max_len_trace = max_len_trace
        
        # Number of samples in batch
        self.batch_size = batch_size

        # Add DQN to this class
        self.Policy = Policy

    def get_batches(self, batch):
        
        s_batch, a_batch, r_batch = [], [], []
        for i in range(len(batch)):
            s_batch.append(list(np.array(batch[i])[:,0]))
            a_batch.append(list(np.array(batch[i])[:,1]))
            r_batch.append(list(np.array(batch[i])[:,2]))
        
        return s_batch, a_batch, r_batch
    
    def train(self, env):
        
        # Store the accumulated reward for each step
        rewards = []
        episode_reward = 0

        # Progress bar to update during while loop
        pbar = tqdm(total=self.budget)

        # Counter
        step = 0
        # Initialize the start state
        s = env.reset()
        trace, batch = [], []
        
        while step < self.budget:
            step += 1
            pbar.update(1)
            
            # Select an action using the policy
            a = self.Policy.select_action(s[None,:])
            
            # Simulate the environment
            s_next, r, done, info = env.step(a)
            
            # Store observations in trace
            trace.append([s, a, r, s_next, done])
            
            # Update the reward of the current episode
            #episode_reward += r
            
            if done or (len(trace) == self.max_len_trace):
                # Reset the environment
                s = env.reset()
                
                # Store trace in batch and clear trace
                batch.append(trace)
                trace = []
                
            if (len(batch) == self.batch_size):
                # Sufficient traces stored, apply update
                
                # Create batch arrays
                s_batch, a_batch, r_batch = self.get_batches(batch)
                
                # Apply update
                self.Policy.update(s_batch, a_batch, r_batch)
                
                # Record the total reward for each trace
                r_batch_sum = [np.sum(r_batch_i) for r_batch_i in r_batch]
                # Average over the traces
                rewards.append(np.mean(r_batch_sum))
                print(f'Average reward of previous {self.batch_size} traces: {rewards[-1]}')
                
                # Clear the batch
                batch = []
                
            

if __name__ == '__main__':
    
    # Create the environment
    env = gym.make("CartPole-v1")

    # Create the policy class
    policy = Policy()
    
    # Create the agent class
    agent = Agent(Policy=policy)
    
    print('\n'*6)
    agent.train(env)