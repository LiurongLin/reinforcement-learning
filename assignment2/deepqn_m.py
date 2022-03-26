import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import argparse
from multiprocessing import Pool

from tqdm import tqdm

from Helper import softmax, argmax, linear_anneal, smooth

class DQN:
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.01, gamma=0.9, architecture='32', with_ep=True,
                 max_replay_buffer_size=1000, replay_buffer_batch_size=64, with_target_network=True):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.8

        # Initialize the Q network
        self.build_Qnet(architecture)

        # Utilise Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Utilise MSE loss
        self.loss_function = tf.keras.losses.MSE

        # Experience replay
        self.with_ep = with_ep
        self.replay_buffer = []
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer_batch_size = replay_buffer_batch_size

        # Target network
        self.with_target_network = with_target_network
        self.Qnet_target = tf.keras.models.clone_model(self.Qnet)

    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):

        # Retrieve the Q-value for each action in state s
        Q_s = self.Qnet.predict(s[None, :])

        if policy == 'egreedy':
            # Epsilon-greedy policy
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() <= epsilon:
                # Select a random action
                a = np.random.randint(0, self.n_actions)
            else:
                # Use greedy policy
                a = argmax(Q_s[0])

        elif policy == 'softmax':
            # Boltzmann policy
            if temp is None:
                raise KeyError("Provide a temperature")

            # Sample action with probability from softmax
            prob = softmax(Q_s[0], temp)
            a = np.random.choice(self.n_actions, p=prob)

        return a

    def build_Qnet(self, arc='32'):

        # Initialize the Q network
        self.Qnet = tf.keras.Sequential()

        # Input-shape is 4: [cart position, cart velocity,
        # pole angle, pole angular velocity]
        self.Qnet.add(tf.keras.Input(shape=self.state_shape))

        # Densely-connected layers
        if arc == '32':
            self.Qnet.add(tf.keras.layers.Dense(32, activation='relu'))
        elif arc == '64':
            self.Qnet.add(tf.keras.layers.Dense(64, activation='relu'))
        elif arc == '32_32':
            self.Qnet.add(tf.keras.layers.Dense(32, activation='relu'))
            self.Qnet.add(tf.keras.layers.Dense(32, activation='relu'))
        else:
            raise ValueError(f'Architecture {arc} unknown')

        # Outputs the expected reward for each action
        self.Qnet.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))

        # Show the architecture of the NN
        # self.Qnet.summary()

    def add_to_replay_buffer(self, s, a, s_next, r, done):

        # Remove the first element of the replay buffer if it has the maximum size
        if len(self.replay_buffer) >= self.max_replay_buffer_size:
            self.replay_buffer.pop(0)

        # Add the step variables to the replay buffer
        self.replay_buffer.append((s, a, r, s_next, done))

    def update_target_network(self):
        # Copy Qnet parameters to the target Qnet
        self.Qnet_target.set_weights(self.Qnet.get_weights())

    def sample_replay_buffer(self):
        # Retrieve a sample of data
        replay_buffer_size = len(self.replay_buffer)
        sample_idx = np.random.randint(0, replay_buffer_size, 
                                       size=min(self.replay_buffer_batch_size, replay_buffer_size))
        # Map the mini-batch to separate arrays
        mini_batch = map(np.array, zip(*[self.replay_buffer[i] for i in sample_idx]))
        return mini_batch

    def update(self, s_batch, a_batch, s_next_batch, r_batch, done_batch):

        with tf.GradientTape() as tape:
            tape.watch(self.Qnet.trainable_variables)

            # Determine what the Q-values are for state s and s_next
            Q_s = self.Qnet(s_batch)
            # print('Q_s in update function:', Q_s[0])

            if self.with_target_network:
                Q_s_next = self.Qnet_target(s_next_batch)
            else:
                Q_s_next = self.Qnet(s_next_batch)

            # New back-up estimate / target
            # If done, done_batch[i]==1 so target_q=r
            target_q = r_batch + (1 - done_batch) * self.gamma * np.max(Q_s_next, axis=1)

            # Actual q value taking action a
            predicted_q = tf.reduce_sum(tf.one_hot(a_batch, self.n_actions) * Q_s, axis=1)

            # Calculate the loss
            loss = self.loss_function(target_q, predicted_q)

        # Backpropagate the gradient to the network's weights
        gradients = tape.gradient(loss, self.Qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Qnet.trainable_variables))

    
def train_Qnet(env, DQN, budget=50000, policy='egreedy', epsilon=0.8, temp=None, with_decay=True, target_update_step=50):
    
    # Store the accumulated reward for each step
    rewards = []

    # Progress bar to update during while loop
    pbar = tqdm(total=budget)

    # Counters
    step = 0

    eps = epsilon
    t = temp

    episode_reward = 0
    # Initialize the start state
    s = env.reset()
    
    while step < budget:
        
        step += 1
        pbar.update(1)
        
        # Create a target network every N steps
        if DQN.with_target_network and (step % target_update_step == 0):
            DQN.update_target_network()

        if with_decay:
            # Use a decaying epsilon/tau
            if policy == 'egreedy':
                eps = linear_anneal(t=step, T=budget, start=0.8, final=0.01, percentage=0.9)
            elif policy == 'softmax':
                t = linear_anneal(t=step, T=budget, start=20, final=0.05, percentage=0.9)
                
        # Select an action using the policy
        a = DQN.select_action(s, policy=policy, epsilon=eps, temp=t)
        
        # Simulate the environment
        s_next, r, done, info = env.step(a)
        
        # Update the reward of the current episode
        episode_reward += r
        
        if DQN.with_ep:
            # Store in the replay buffer
            DQN.add_to_replay_buffer(s, a, s_next, r, done)

            # Sample a batch from the replay buffer
            s_batch, a_batch, r_batch, s_next_batch, done_batch = DQN.sample_replay_buffer()
        else:
            # Reshape the data
            s_batch, a_batch, r_batch, s_next_batch, done_batch = s[None, :], np.array(a), np.array(r), \
                                                                  s_next[None, :], np.array(done)
        
        # Update the Q network
        DQN.update(s_batch, a_batch, s_next_batch, r_batch, done_batch)
            
        if done:
            # Collect the total reward of the episode
            rewards.append(episode_reward)

            # Reset the episode reward
            episode_reward = 0
        
            # Initialize the start state
            s = env.reset()
        else:
            # Update the state
            s = s_next

            if step == budget:
                rewards.append(episode_reward)

    return rewards


def test(args_dict):

    # Create the environment
    env = gym.make("CartPole-v1")

    # Create the deep Q network
    net = DQN(with_ep=args_dict['experience_replay'], 
              with_target_network=args_dict['target_network'],
              learning_rate=args_dict['learning_rate']
             )

    rewards = train_Qnet(env, net, 
                         budget=args_dict['budget'], 
                         policy=args_dict['policy'],
                         epsilon=args_dict['epsilon'],
                         temp=args_dict['temp'],
                         with_decay=args_dict['with_decay'],
                         target_update_step=args_dict['target_update_step']
                         )


    return rewards


def episode_rewards_to_save_array(rewards):
    """
    Converts the episode rewards, which is a list of lists, to one single array, which can be saved by numpy
    The 'rewards' variable contains a list for each repetition, which contains a total reward for each episode
    """
    rewards_per_rep_arr = []
    for repetition in rewards:
        # Use a 0 as a delimiter between repetitions
        rewards_per_rep_arr += repetition + [0]
    rewards_per_rep_arr = np.array(rewards_per_rep_arr, dtype=np.float32)
    return rewards_per_rep_arr


def read_arguments():
    parser = argparse.ArgumentParser()

    # All arguments to expect
    parser.add_argument('--experience_replay', nargs='?', const=True, default=False, help='Use experience replay')
    parser.add_argument('--buffer_size', nargs='?', type=int, default=100, help='Size of the experience replay buffer')

    parser.add_argument('--target_network', nargs='?', const=True, default=False, help='Use target network')
    parser.add_argument('--target_update_step', nargs='?', type=int, default=50,
                        help='Number of steps between updates of target network')

    parser.add_argument('--budget', nargs='?', type=int, default=10000, help='Total number of steps')
    parser.add_argument('--n_repetitions', nargs='?', type=int, default=8, help='Number of repetitions')
    parser.add_argument('--n_cores', nargs='?', type=int, default=4, help='Number of cores to divide repetitions over')

    parser.add_argument('--policy', nargs='?', type=str, default='egreedy', help='Policy to use (egreedy/softmax)')
    parser.add_argument('--epsilon', nargs='?', type=float, default=0.2, help='Epsilon exploration parameter')
    parser.add_argument('--temp', nargs='?', type=float, default=1, help='Tau (temperature) exploration parameter')

    parser.add_argument('--with_decay', nargs='?', const=True, default=False, help='Use decaying exploration parameter')

    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.01, help='Learing rate')

    parser.add_argument('--architecture', nargs='?', type=str, default='32', help='Shape(s) of the hidden layer(s)')

    # Read the arguments in the command line
    args = parser.parse_args()

    args_dict = vars(args)  # Create a dictionary

    return args_dict


if __name__ == '__main__':

    # Use 5 for loops for hyperparameter tuning

    # lr = [0.01, 0.005, 0.001] (3)
    # bs = [10, 100, 1000] (3)
    # tus = [10, 100, 1000] (3)
    # eps = [0.2, 0.1, 0.05] + 'with_decay', t = [10, 1, 0.1] + 'with_decay' (8)
    # arc = ['32', '64', '32_32'] (3)
    
    print()
    args_dict = read_arguments()
    print(args_dict)

    # args_dict is placed in a list to avoid unpacking the dictionary
    params = args_dict['n_repetitions'] * [[args_dict]]

    # Run the repetitions on multiple cores
    pool = Pool(args_dict['n_cores'])
    rewards_per_rep = pool.starmap(test, params)
    pool.close()
    pool.join()

    save_arr = episode_rewards_to_save_array(rewards_per_rep)

    # generate the file name
    filename = "we={}_wtn={}_arc={}_lr={}_tus={}_bs={}_pol={}_eps={}_t={}_wd={}.npy".format(args_dict['experience_replay'],
                                                                                            args_dict['target_network'],
                                                                                            args_dict['architecture'],
                                                                                            args_dict['learning_rate'],
                                                                                            args_dict['target_update_step'],
                                                                                            args_dict['buffer_size'],
                                                                                            args_dict['policy'],
                                                                                            args_dict['epsilon'],
                                                                                            args_dict['temp'],
                                                                                            args_dict['with_decay'])

    # Save the total reward of every episode of every repetition
    results_dir = './results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    path = os.path.join(results_dir, filename)
    np.save(path, save_arr)

