import os
import matplotlib.pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

from Helper import softmax, argmax, linear_anneal, smooth


class DQN:
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.001, gamma=0.9,
                 with_replay=True, max_replay_buffer_size=1000,
                 replay_batch_size=50, with_target_network=True, target_update_step=8):

        self.state_shape = state_shape
        self.n_actions = n_actions
        # self.architecture = [int(node_i) for node_i in architecture.split('_')]
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize the Q network
        self.Qnet = self.build_Qnet()

        # Utilise Adam optimizer
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Utilise MSE loss
        #self.loss_function = tf.keras.losses.MSE

        # Experience replay
        self.with_replay = with_replay
        self.replay_buffer = []
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_batch_size = replay_batch_size

        # Target network
        self.with_target_network = with_target_network
        self.Qnet_target = tf.keras.models.clone_model(self.Qnet)
        self.target_update_step = target_update_step

    def build_Qnet(self):

        # Initialize the Q network
        X_input = Input(self.state_shape)

        # 'Dense' is the basic form of a neural network layer
        # Input Layer of state size(4) and Hidden Layer with 512 nodes
        X = Dense(512, input_shape=self.state_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

        # Hidden layer with 256 nodes
        X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

        # Hidden layer with 64 nodes
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(self.n_actions, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X)
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        model.summary()
        return model

        # Show the architecture of the NN
        # self.Qnet.summary()

    def add_to_replay_buffer(self, s, a, s_next, r, done):
        # Remove the first element of the replay buffer if it has the maximum size
        if len(self.replay_buffer) >= self.max_replay_buffer_size:
            self.replay_buffer.pop(0)

        # Add the step variables to the replay buffer
        self.replay_buffer.append((s, a, r, s_next, done))

    def sample_replay_buffer(self):
        # Retrieve a sample of data
        replay_buffer_size = len(self.replay_buffer)
        sample_idx = np.random.randint(0, replay_buffer_size,
                                       size=min(self.replay_batch_size, replay_buffer_size))

        # Map the mini-batch to separate arrays
        mini_batch = map(np.array, zip(*[self.replay_buffer[i] for i in sample_idx]))
        return mini_batch

    def update_target_network(self):
        # Copy Qnet parameters to the target Qnet
        self.Qnet_target.set_weights(self.Qnet.get_weights())


class Agent:
    def __init__(self, DQN, episodes = 500, policy = 'egreedy', gamma = 0.95):

        # Number of timesteps to train
        self.episodes = episodes

        # Exploration strategy
        self.policy = policy

        #annealing epsilon parameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999

        # annealing temp parameters
        self.temp = 1e4  # exploration rate
        self.temp_min = 0.5
        self.temp_decay = 0.8

        #step to start learning
        self.start = 1000
        self.gamma = gamma


        # Add DQN to this class
        self.DQN = DQN

    def select_action(self, s):

        # Retrieve the Q-value for each action in state s
        Q_s = self.DQN.Qnet.predict(s[None, :])

        if self.policy == 'egreedy':
            # Epsilon-greedy policy
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            if np.random.rand() <= self.epsilon:
                # Select a random action
                a = np.random.randint(0, self.DQN.n_actions)
            else:
                # Use greedy policy
                a = argmax(Q_s[0])

        elif self.policy == 'softmax':
            # Boltzmann policy
            if self.temp is None:
                raise KeyError("Provide a temperature")

            # Sample action with probability from softmax
            prob = softmax(Q_s[0], self.temp)
            a = np.random.choice(self.DQN.n_actions, p=prob)

        return a

    def update(self, s_batch, a_batch, s_next_batch, r_batch, done_batch):


        # Determine the Q-values for state s
        Q_s = self.DQN.Qnet.predict(s_batch)

        # Determine the Q-values for future state
        if self.DQN.with_target_network:
            Q_s_next = self.DQN.Qnet_target.predict(s_next_batch)
        else:
            Q_s_next = self.DQN.Qnet.predict(s_next_batch)


        if self.DQN.with_replay:
            for i in range (len(a_batch)):
                if done_batch[i]:
                    Q_s[i][a_batch[i]] = r_batch[i]
                else:
                    # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    Q_s[i][a_batch[i]] = r_batch[i] + self.gamma * (np.amax(Q_s_next[i]))
            self.DQN.Qnet.fit(s_batch, Q_s, verbose=0)
        else:
            if done_batch:
                Q_s[0][a_batch] = r_batch
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                Q_s[0][a_batch] = r_batch + self.gamma * (np.amax(Q_s_next[0]))
            self.DQN.Qnet.fit(s_batch, Q_s, verbose=0)

    def train(self, env):

        # Store the accumulated reward for each step
        rewards = []
        episode_reward = 0

        # Progress bar to update during while loop
        #pbar = tqdm(total=self.budget)

        # Counter
        step_all = 0
        # Initialize the start state
        episode = 0

        while episode < self.episodes:
            s = env.reset()
            done = False
            step = 0
            while not done:

                # Select an action using the policy
                a = self.select_action(s)

                # Simulate the environment
                s_next, r, done, info = env.step(a)

                if not done or step == env._max_episode_steps - 1:
                    r = r
                else:
                    r = -100

                if self.DQN.with_replay:
                    # Store in the replay buffer
                    self.DQN.add_to_replay_buffer(s, a, s_next, r, done)

                    # Sample a batch from the replay buffer
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = self.DQN.sample_replay_buffer()
                else:
                    # Reshape the data
                    s_batch, a_batch, r_batch, s_next_batch, done_batch = s[None, :], np.array(a), np.array(r), \
                                                                                      s_next[None, :], np.array(done)

                if step_all > self.start:
                    if self.policy == 'egreedy':
                        if self.epsilon > self.epsilon_min:
                            self.epsilon *= self.epsilon_decay

                    else:
                        if self.temp > self.temp_min:
                            self.temp *= self.temp_decay

                s = s_next
                step += 1
                step_all += 1
                if done:
                    if self.policy == 'egreedy':
                        print("episode: {}/{}, score: {}, e: {:.2}".format(episode, self.episodes, step, self.epsilon))
                    else:
                        print("episode: {}/{}, score: {}, t: {:.2}".format(episode, self.episodes, step, self.temp))
                    rewards.append(step)
                # update the Q network
                self.update(s_batch, a_batch, s_next_batch, r_batch, done_batch)
            if self.DQN.with_target_network and (episode % self.DQN.target_update_step == 0):
                self.DQN.update_target_network()

            episode += 1

        return rewards



if __name__ == "__main__":
    rewards_list = []
    for i in range (4):
        env = gym.make("CartPole-v1")
        dqn = DQN(with_target_network=True, with_replay=True)
        agent = Agent(DQN = dqn, policy = 'egreedy')
        rewards = agent.train(env)
        plt.plot(smooth(rewards,8), alpha = 0.25)
        rewards_list.append(smooth(rewards,8))
    mean_r = np.mean(rewards_list, axis = 0)
    plt.plot(mean_r)
    plt.xlabel('Episode')
    plt.ylabel('rewards')
    plt.savefig("+tn+er_500.png")
    plt.show()