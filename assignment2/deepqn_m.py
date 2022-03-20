import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import argparse
from multiprocessing import Pool

from tqdm import tqdm

from Helper import softmax, argmax, linear_anneal, smooth


class DQN:
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.01, gamma=0.9, with_ep=True,
                 max_replay_buffer_size=1000, replay_buffer_batch_size=64, with_target_network=True):

        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 0.8

        # Initialize the Q network
        self.build_Qnet()

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

    def build_Qnet(self):

        # Initialize the Q network
        self.Qnet = tf.keras.Sequential()

        # Input-shape is 4: [cart position, cart velocity,
        # pole angle, pole angular velocity]
        self.Qnet.add(tf.keras.Input(shape=self.state_shape))

        # Densely-connected layers
        self.Qnet.add(tf.keras.layers.Dense(32, activation='relu'))

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
        sample_idx = np.random.randint(0, replay_buffer_size, size=min(self.replay_buffer_batch_size, replay_buffer_size))
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


def plot_rewards(rewards, config_labels, save_file=None, ):
    n_steps = len(rewards[0])
    steps = np.arange(n_steps)
    smoothing_window = n_steps // 10 + 1
    n_configs = len(config_labels)

    # smooth_rewards = [smooth(r, len(rewards) // 10 + 1) for r in rewards]
    # mean_rewards = np.mean(smooth_rewards, axis=0)
    # std_rewards = np.std(smooth_rewards, axis=0)
    # upper = mean_rewards + std_rewards
    # lower = mean_rewards - std_rewards

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i in range(n_configs):
        config_rewards = np.mean(np.array(rewards)[i::n_configs], axis=0)
        ax.plot(steps, smooth(config_rewards, smoothing_window), label=config_labels[i])
    # ax.fill_between(steps, upper, lower, alpha=0.2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean reward')
    ax.set_title('DQN mean reward progression')
    ax.legend()
    if save_file is not None:
        plt.savefig(save_file)
    plt.show()
    plt.close()


def train_Qnet(env, DQN, budget=50000, policy='egreedy', epsilon=0.8, temp=None, with_decay=True):
    # Store the accumulated reward for each step
    rewards = []

    # Progress bar to update during while loop
    pbar = tqdm(total=budget)

    # Counters
    step = 0

    eps = epsilon
    t = temp

    while step < budget:

        done = False

        # Initialize the start state
        s = env.reset()
        episode_reward = 0

        # Create a target network every 100 steps
        if DQN.with_target_network and (step % 500 == 0):
            DQN.update_target_network()

        if with_decay:
            # Use a decaying epsilon/tau
            if policy == 'egreedy':
                eps = linear_anneal(t=step, T=budget, start=epsilon, final=0.01, percentage=0.9)
            elif policy == 'softmax':
                t = linear_anneal(t=step, T=budget, start=temp, final=0.01, percentage=0.9)

        while not done and step < budget:
            a = DQN.select_action(s, policy=policy, epsilon=eps, temp=t)

            # Simulate the environment
            s_next, r, done, info = env.step(a)
            step += 1
            pbar.update(1)

            # collect the rewards
            episode_reward += r
            rewards.append(episode_reward)

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
                break
            else:
                s = s_next  # Update the state

    return rewards


def test(budget, with_ep=False, with_target_network=False):
    env = gym.make("CartPole-v1")

    # Create the deep Q network
    net = DQN(with_ep=with_ep, with_target_network=with_target_network)

    rewards = train_Qnet(env, net, budget=budget, with_decay=False)
    return rewards


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--experience_replay', nargs='?', const=True, default=False, help="Use experience replay")
parser.add_argument('-t', '--target_network', nargs='?', const=True, default=False, help="Use target network")

if __name__ == '__main__':
    args = parser.parse_args()

    n_repititions = 8
    n_budget = 10000
    n_cores = 4

    # params = [(n_budget, args.experience_replay, args.target_network) for _ in range(n_repititions)]
    params = n_repititions * [(n_budget, False, False),
                              (n_budget, True, False),
                              (n_budget, False, True),
                              (n_budget, True, True)]
    pool = Pool(n_cores)
    rewards_per_rep = pool.starmap(test, params)
    pool.close()
    pool.join()

    labels = ['dqn', 'dqn with ep', 'dqn with tn', 'dqn with ep and tn']
    plot_rewards(rewards_per_rep, config_labels=labels, save_file='dqn_rewards')
