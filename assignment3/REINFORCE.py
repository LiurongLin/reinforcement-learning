import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm


class Policy:
    def __init__(self, state_shape=(4,), n_actions=2, lr=1e-3, gamma=0.9, actor_arc=(64, 64), actor_activation=None,
                 critic_arc=(64, 64), critic_activation=None, with_bootstrap=True, with_baseline=True):
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.gamma = gamma
        self.lr  = lr
        self.actor_arc = actor_arc
        if actor_activation is None:
            self.actor_activation = ['relu']*len(self.actor_arc)
        else:
            self.actor_activation = actor_activation

        self.critic_arc = critic_arc
        if actor_activation is None:
            self.critic_activation = ['relu'] * len(self.critic_arc)
        else:
            self.critic_activation = critic_activation

        self.with_bootstrap = with_bootstrap
        self.with_baseline = with_baseline
        
        # Initialize neural network
        self.build_actor()
        if self.with_bootstrap or self.with_baseline:
            self.build_critic()
        
        # Use Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
   
    def build_actor(self):
        # Initialize neural network
        self.actor = tf.keras.Sequential()
        
        # Input layer with shape (4,)
        self.actor.add(tf.keras.Input(shape=self.state_shape))
        
        # Densely-connected layers
        for nodes_i, activation_i in zip(self.actor_arc, self.actor_activation):
            self.actor.add(tf.keras.layers.Dense(nodes_i, activation=activation_i))
        
        # Output layer with softmax activation, probability of taking each action
        self.actor.add(tf.keras.layers.Dense(self.n_actions, activation='softmax'))
        
        # Show the architecture of the NN
        # self.NN.summary()

    def build_critic(self):
        # Initialize neural network
        self.critic = tf.keras.Sequential()

        # Input layer with shape (4,)
        self.critic.add(tf.keras.Input(shape=self.state_shape))

        # Densely-connected layers
        for nodes_i, activation_i in zip(self.critic_arc, self.critic_activation):
            self.critic.add(tf.keras.layers.Dense(nodes_i, activation=activation_i))

        # Output layer with softmax activation, probability of taking each action
        self.critic.add(tf.keras.layers.Dense(1))

        # Show the architecture of the NN
        # self.NN.summary()
    
    def select_action(self, s):
        
        # Probability of taking each action
        prob_s = np.array(self.actor(s)).flatten()
        
        # Add some noise for exploration
        #prob_s = ...
        
        # Sample action
        a = np.random.choice(self.n_actions, p=prob_s)
        return a
    
    def loss_function(self, s_batch, a_batch, r_batch, s_next_batch):

        N_traces = len(r_batch)

        actor_loss = tf.constant(0, dtype=tf.float32)
        critic_loss = tf.constant(0, dtype=tf.float32)
        for trace_idx in range(N_traces):
            # Loop over each trace in batch
            s_in_trace = tf.convert_to_tensor(s_batch[trace_idx])
            a_in_trace = a_batch[trace_idx]
            r_in_trace = r_batch[trace_idx]
            s_next_in_trace = s_next_batch[trace_idx]

            len_trace = len(r_in_trace)

            # Predict the probabilities for each action
            prob_s = self.actor(s_in_trace)

            # Probabilities for chosen actions
            prob_sa = tf.reduce_sum(tf.one_hot(a_in_trace, self.n_actions)*prob_s, axis=1)

            Q_sa = 0
            for step in range(len_trace)[::-1]:
                # Loop backwards over each observation in trace

                if self.with_bootstrap:
                    V_s_next = self.critic(s_next_in_trace[step][None, ...])[0, 0]
                else:
                    V_s_next = self.gamma * Q_sa

                Q_sa = r_in_trace[step] + V_s_next
                A_sa = Q_sa - V_s_next

                if self.with_baseline:
                    actor_loss += A_sa * tf.math.log(prob_sa[step])
                else:
                    actor_loss += Q_sa * tf.math.log(prob_sa[step])

                critic_loss += (A_sa)**2

        # Multiply by -1 to create a loss function
        actor_loss = - 1/N_traces * actor_loss
        critic_loss = - 1/N_traces * critic_loss
        return actor_loss + critic_loss

    def update(self, s_batch, a_batch, r_batch, s_next_batch):

        if self.with_bootstrap:
            train_vars = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            train_vars = self.actor.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(train_vars)

            # Calculate the loss
            loss_value = self.loss_function(s_batch, a_batch, r_batch, s_next_batch)

        print(' ')
        print('loss_value', loss_value.numpy())

        gradients = tape.gradient(loss_value, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))


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
        
        s_batch, a_batch, r_batch, s_next_batch = [], [], [], []
        for i in range(len(batch)):
            s_batch.append(list(np.array(batch[i])[:,0]))
            a_batch.append(list(np.array(batch[i])[:,1]))
            r_batch.append(list(np.array(batch[i])[:,2]))
            s_next_batch.append(list(np.array(batch[i])[:,3]))

        return s_batch, a_batch, r_batch, s_next_batch
    
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
                
            if len(batch) == self.batch_size:
                # Sufficient traces stored, apply update
                
                # Create batch arrays
                s_batch, a_batch, r_batch, s_next_batch = self.get_batches(batch)
                
                # Apply update
                self.Policy.update(s_batch, a_batch, r_batch, s_next_batch)
                
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
