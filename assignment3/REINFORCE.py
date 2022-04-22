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
                 critic_arc=(64, 64), critic_activation=None, n=1, with_entropy=True, eta=0.01, with_bootstrap=True, with_baseline=True):
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        self.gamma = gamma
        self.lr  = lr
        self.eta = eta
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
        self.with_entropy = with_entropy
        
        # Initialize neural network
        self.build_actor()
        self.build_critic()

        self.n = n
        
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
        
    def entropy(self, p):
        return -tf.reduce_sum(p * tf.math.log(p) / tf.math.log(2.), axis=0)
    
    def loss_function(self, s_batch, a_batch, r_batch):

        N_traces = len(r_batch)

        obj = tf.constant(0, dtype=tf.float32)
        gamma = tf.constant([self.gamma], dtype=tf.float32)
        for trace_idx in range(N_traces):
            # Loop over each trace in batch
            s_in_trace = tf.convert_to_tensor(s_batch[trace_idx])
            a_in_trace = a_batch[trace_idx]
            r_in_trace = tf.convert_to_tensor(r_batch[trace_idx])

            episode_length = len(r_in_trace)

            # Predict the probabilities for each action
            prob_s = self.actor(s_in_trace)

            # Probabilities for chosen actions
            prob_sa = tf.reduce_sum(tf.one_hot(a_in_trace, self.n_actions)*prob_s, axis=1)

            rewards = r_in_trace * tf.pow(gamma, tf.range(episode_length, dtype=tf.float32))
            V = self.critic(s_in_trace)[:, 0]
            
            n = min(self.n, episode_length - 1)
            for t in range(episode_length - n):
                if self.with_bootstrap:
                    V_sn = self.gamma**(t + n) * V[t + n]
                    Q_sa = tf.reduce_sum(rewards[t:t + n], axis=0) + V_sn
                else:
                    Q_sa = tf.reduce_sum(rewards[t:], axis=0)

                A_sa = Q_sa - V[t]

                if self.with_baseline:
                    Psi_t = A_sa
                else:
                    Psi_t = Q_sa
                
                if self.with_entropy:
                    actor_obj = Psi_t * tf.math.log(prob_sa[t]) + self.eta * self.entropy(prob_s[t])
                else:
                    actor_obj = Psi_t * tf.math.log(prob_sa[t])
                
                if self.with_bootstrap or self.with_baseline:
                    critic_loss = A_sa**2
                    obj += actor_obj - critic_loss
                else:
                    obj += actor_obj

        # Multiply by -1 to create a loss function
        loss = - 1/N_traces * obj
        return loss

    def update(self, s_batch, a_batch, r_batch):

        if self.with_bootstrap:
            train_vars = self.actor.trainable_variables + self.critic.trainable_variables
        else:
            train_vars = self.actor.trainable_variables

        with tf.GradientTape() as tape:
            tape.watch(train_vars)

            # Calculate the loss
            loss_value = self.loss_function(s_batch, a_batch, r_batch)

        print(' ')
        print('loss_value', loss_value.numpy())

        gradients = tape.gradient(loss_value, train_vars)
        self.optimizer.apply_gradients(zip(gradients, train_vars))


class Agent:
    def __init__(self, Policy, budget=50000, max_len_trace=500, batch_size=5):
        
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
            s_batch.append(list(np.array(batch[i])[:, 0]))
            a_batch.append(list(np.array(batch[i])[:, 1]))
            r_batch.append(list(np.array(batch[i])[:, 2]))

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
            a = self.Policy.select_action(s[None, :])
            
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
            else:
                s = s_next
                
            if len(batch) == self.batch_size:
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
                print(' ')
                
                # Clear the batch
                batch = []


        filename = "boostrap={}_baseline={}_entropy={}.npy".format(self.Policy.with_bootstrap,
                                                                   self.Policy.with_baseline,
                                                                   self.Policy.with_entropy)
        np.save(filename, rewards)
        plt.plot(rewards)
        plt.show()
                

def pool_function(args_dict):
    # Create the environment
    env = gym.make("CartPole-v1")

    # Create the policy class
    policy = Policy(lr=args_dict['lr'], 
                    actor_arc=args_dict['actor_arc'], 
                    critic_arc=args_dict['critic_arc'], 
                    n=args_dict['n_boot'], 
                    with_bootstrap=args_dict['with_bootstrap'], 
                    with_baseline=args_dict['with_baseline'],
                    with_entropy=args_dict['with_entropy'],
                    eta=args_dict['eta'],
                   )
    
    # Create the agent class
    agent = Agent(Policy=policy,
                  budget=args_dict['budget']
                 )
    
    print('\n'*3)
    agent.train(env)
    
        
def read_arguments():
    parser = argparse.ArgumentParser()

    # All arguments to expect
    parser.add_argument('--with_bootstrap', nargs='?', const=True, default=False, 
                        help='Use bootstrapping')
    parser.add_argument('--n_boot', nargs='?', type=int, default=1, 
                        help='Number of values to bootstrap over')
    
    parser.add_argument('--with_baseline', nargs='?', const=True, default=False, 
                        help='Use baseline')
    
    parser.add_argument('--with_entropy', nargs='?', const=True, default=False, 
                        help='Use entropy regularization')
    parser.add_argument('--eta', nargs='?', type=float, default=0.01,
                        help='Temperature parameter to scale the entropy regularization')
    
    parser.add_argument('--lr', nargs='?', type=float, default=1e-3, 
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--actor_arc', nargs='?', type=tuple, default=(64,64), 
                        help='Shape(s) of the hidden layer(s) for the actor network')
    parser.add_argument('--critic_arc', nargs='?', type=tuple, default=(64,64), 
                        help='Shape(s) of the hidden layer(s) for the critic network')

    parser.add_argument('--budget', nargs='?', type=int, default=50000, 
                        help='Total number of steps')
    parser.add_argument('--n_repetitions', nargs='?', type=int, default=1, 
                        help='Number of repetitions')
    parser.add_argument('--n_cores', nargs='?', type=int, default=1, 
                        help='Number of cores to divide repetitions over')
    
    parser.add_argument('--results_dir', nargs='?', type=str, default='./results', 
                        help='Directory to store the results in')

    # Read the arguments in the command line
    args = parser.parse_args()

    args_dict = vars(args)  # Create a dictionary

    return args_dict


if __name__ == '__main__':
    
    # Read the arguments in the command line
    args_dict = read_arguments()

    print('\nSupplied arguments:')
    for key in args_dict.keys():
        print(f'{key}:', args_dict[key])
    print('\n\n')


    # args_dict is placed in a list to avoid unpacking the dictionary
    params = args_dict['n_repetitions'] * [[args_dict]]

    # Run the repetitions on multiple cores
    pool = Pool(args_dict['n_cores'])
    rewards_per_rep = pool.starmap(pool_function, params)
    pool.close()
    pool.join()

    # Save the rewards to a .npy file
    #save_rewards(args_dict, rewards_per_rep)
