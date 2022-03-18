import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

from tqdm import tqdm

from Helper import softmax, argmax, linear_anneal


class DQN:
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.01, gamma=0.9, with_ep=True,
                 max_replay_buffer_size=1000, with_target_network=True):
        
        self.state_shape   = state_shape
        self.n_actions     = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Initialize the Q network
        self.build_Qnet()
        
        # Utilise Adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Utilise MSE loss
        self.loss_function = tf.keras.losses.MSE

        # Experience replay
        self.with_ep = with_ep
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer = []
        
        # Use a target network
        self.with_target_network = with_target_network

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
        self.Qnet.add(tf.keras.layers.Dense(64, activation='relu'))
        self.Qnet.add(tf.keras.layers.Dense(32, activation='relu'))

        # Outputs the expected reward for each action
        self.Qnet.add(tf.keras.layers.Dense(self.n_actions, activation='linear'))

        # Show the architecture of the NN
        self.Qnet.summary()
    
    def add_to_replay_buffer(self, s, a, s_next, r, done):
        
        if len(self.replay_buffer) >= self.max_replay_buffer_size:
            self.replay_buffer.pop(0)
            
        self.replay_buffer.append((s, a, r, s_next, done))
        
    def update(self, s_batch, a_batch, s_next_batch, r_batch, done_batch):
                
        with tf.GradientTape() as tape:
            tape.watch(self.Qnet.trainable_variables)
            
            # Determine what the Q-values are for state s and s_next
            Q_s = self.Qnet(s_batch)
            #print('Q_s in update function:', Q_s[0])
            
            if self.with_target_network:
                Q_s_next = self.Qnet_target(s_next_batch)
            else:
                Q_s_next = self.Qnet(s_next_batch)
            
            # New back-up estimate / target
            # If done, done_batch[i]==1 so target_q=r
            target_q = r_batch + (1-done_batch)*self.gamma*np.max(Q_s_next, axis=1)

            # Actual q value taking action a
            predicted_q = tf.reduce_sum(tf.one_hot(a_batch, self.n_actions) * Q_s, axis=1)

            # Calculate the loss
            loss = self.loss_function(target_q, predicted_q)
        
        # Backpropagate the gradient to the network's weights
        gradients = tape.gradient(loss, self.Qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Qnet.trainable_variables))
        
def train_Qnet(env, DQN, N_episodes=500, policy='egreedy', epsilon=0.8, temp=None, with_decay=True):

    returns = []

    for episode in tqdm(range(N_episodes)):
        
        done = False

        # Initialize the start state
        s, info = env.reset(seed=42, return_info=True)

        rewards = 0

        # Create a target network every 20 episodes
        if DQN.with_target_network and (episode%20 == 0):
            DQN.Qnet_target = tf.keras.models.clone_model(DQN.Qnet)
            DQN.Qnet_target.set_weights(DQN.Qnet.get_weights()) 
            
        while not done:
            
            if with_decay:
                # Use a decaying epsilon/tau
                epsilon_new, temp_new = None, None
                if policy=='egreedy':
                    epsilon_new = linear_anneal(t=episode, T=N_episodes, start=epsilon, 
                                                final=0.01, percentage=0.9)
                elif policy=='softmax':
                    temp_new = linear_anneal(t=episode, T=N_episodes, start=temp, 
                                             final=0.01, percentage=0.9)
                # Sample an action with the policy
                a = DQN.select_action(s, policy=policy, epsilon=epsilon_new, temp=temp_new)
            
            else:
                # Select an action with the policy
                a = DQN.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

            # Simulate the environment
            s_next, r, done, info = env.step(a)

            # collect the rewards
            rewards += r
            
            if DQN.with_ep:
                
                # Store in the replay buffer
                DQN.add_to_replay_buffer(s, a, s_next, r, done)
            
                # Retrieve a sample of data
                sample_idx = np.random.randint(0, len(DQN.replay_buffer), 
                                               size=min(64, len(DQN.replay_buffer)))
                mini_batch = np.array(DQN.replay_buffer)[sample_idx]
                
                # Map the mini-batch to separate arrays
                s_batch, a_batch, r_batch, s_next_batch, done_batch = map(np.array, zip(*mini_batch))
                
            else:
                # Reshape the data
                s_batch, a_batch, r_batch, s_next_batch, done_batch = s[None,:], np.array(a), np.array(r), s_next[None,:], np.array(done)

            # Update the Q network
            DQN.update(s_batch, a_batch, s_next_batch, r_batch, done_batch)
            
            if done:
                break
            else:
                s = s_next # Update the state

        returns.append(rewards)
        
        if (episode%100 == 0) and (episode >= 100):
            print('Mean reward of previous 100 timesteps:', np.mean(returns[-100:]))

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
