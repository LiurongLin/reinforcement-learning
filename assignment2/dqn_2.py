import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

from tqdm import tqdm

from Helper import softmax, argmax, linear_anneal


class DQN:
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.0001, gamma=0.1, loss = 'mse',
                 max_replay_buffer_size=1000):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize the Q network for main and target networks
        self.Qnet = self.build_model(loss = loss, lr= learning_rate)
        self.Qnet_target = self.build_model(loss=loss, lr=learning_rate)
        self.Qnet_target.set_weights(self.Qnet.get_weights())
        # # Utilise Adam optimizer
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #
        # # Utilise MSE loss
        # self.loss_function = tf.keras.losses.MSE

        # Experience replay
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer = []
        self.step = 0
        self.target_update_counter = 0

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
                a = argmax(np.squeeze(Q_s))

        elif policy == 'softmax':
            # Boltzmann policy
            if temp is None:
                raise KeyError("Provide a temperature")

            # Sample action with probability from softmax
            prob = softmax(Q_s, temp)
            a = np.random.choice(self.n_actions, p=prob)

        return a

    def build_model(self, loss='mse', lr=0.001):
        '''build a deep learning model'''
        model = tf.keras.Sequential()

        # Input-shape is 4: [cart position, cart velocity,
        # pole angle, pole angular velocity]
        model.add(tf.keras.Input(shape=self.state_shape))

        # Densely-connected layers
        model.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))

        # Outputs the expected reward for each action
        model.add(tf.keras.layers.Dense(self.n_actions, activation='relu'))

        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy'])
        # Show the architecture of the NN
        model.summary()

        return model

    def update(self, s, a, s_next, r, done):

        if len(self.replay_buffer) >= self.max_replay_buffer_size:
            self.replay_buffer.pop(0)



        self.replay_buffer.append((s, a, r, s_next, done))

        sample_idx = np.random.randint(0, min(len(self.replay_buffer), self.max_replay_buffer_size))
        s, a, r, s_next, done = self.replay_buffer[sample_idx]


        # Determine what the Q-values are for state s and s_next using two models
        Q_s = self.Qnet.predict(s[None, :])
        Q_s_next = self.Qnet_target.predict(s_next[None, :])

            # New back-up estimate / target
        if done:
            target_q = r
        else:
            target_q = r + self.gamma * np.max(Q_s_next)

        # Update Q value for a given state and action
        Q_s[:, a] = target_q
        self.Qnet.fit(s[None,:],Q_s, verbose = False)




def train_Qnet(env, DQN, N_episodes=500, policy='egreedy', epsilon=0.2, temp=None):
    returns = []

    for episode in tqdm(range(N_episodes)):

        done = False

        # Initialize the start state
        s, info = env.reset(seed=42, return_info=True)

        rewards = 0

        while not done:
            # Select an action with the policy
            a = DQN.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

            # Simulate the environment
            s_next, r, done, info = env.step(a)

            # collect the rewards
            rewards += r

            # Update the Q network
            DQN.update(s, a, s_next, r, done)

            if done:
                print(rewards)
                break
            else:
                s = s_next  # Update the state

            if episode >= N_episodes * 0.8:
                env.render()

        #set the weights of the target network to be the same as the main network for every 6 episodes.
        DQN.target_update_counter += 1
        if DQN.target_update_counter > 3:
            DQN.Qnet_target.set_weights(DQN.Qnet.get_weights())
            DQN.target_update_counter = 0

        returns.append(rewards)


    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(returns)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title('DQN return progression')
    plt.show()


def test():
    # Create the environment
    env = gym.make("CartPole-v1")

    # Create the deep Q network
    net = DQN()

    train_Qnet(env, net)


if __name__ == '__main__':
    test()