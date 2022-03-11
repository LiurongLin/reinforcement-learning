import tensorflow as tf
import numpy as  np
import gym

from tqdm import tqdm

from Helper import softmax, argmax, linear_anneal

class DQN():
    
    def __init__(self, state_shape=(4,), n_actions=2, learning_rate=0.001, gamma=1.0):
        
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

        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        # Retrieve the Q-value for each action in state s
        Q_s = self.Qnet.predict(s[None,:])
        
        if policy == 'egreedy':
            # Epsilon-greedy policy
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            if np.random.rand() <= epsilon:
                # Select a random action
                a = np.random.randint(0, self.n_actions)
            else:
                # Use greedy policy
                a = argmax(Q_s)
                
        elif policy == 'softmax':
            # Boltzmann policy
            if temp is None:
                raise KeyError("Provide a temperature")
            
            # Sample action with probability from softmax
            prob = softmax(Q_s, temp)
            a = np.random.choice(self.n_actions, p=prob)
            
        return a
    
    def build_Qnet(self):

        # Initialize the Q network
        self.Qnet = tf.keras.Sequential()

        # Input-shape is 4: [cart position, cart velocity, 
        # pole angle, pole angular velocity]
        self.Qnet.add(tf.keras.Input(shape=self.state_shape))

        # Densely-connected layers
        self.Qnet.add(tf.keras.layers.Dense(32, activation='sigmoid'))
        self.Qnet.add(tf.keras.layers.Dense(64, activation='relu'))
        self.Qnet.add(tf.keras.layers.Dense(64, activation='relu'))

        # Outputs the expected reward for each action
        self.Qnet.add(tf.keras.layers.Dense(self.n_actions, activation='relu'))

        
        # Show the architecture of the NN
        self.Qnet.summary()
        
    def update(self, s, a, s_next, r, done):
    
        with tf.GradientTape() as tape:
            tape.watch(self.Qnet.trainable_variables)
            
            # Determine what the Q-values are for state s and s_next
            Q_s      = self.Qnet(s[None,:])
            Q_s_next = self.Qnet(s_next[None,:])

            # New back-up estimate / target
            if done:
                target_q = r
            else:
                target_q = r + self.gamma*np.max(Q_s_next)

            # Actual q value taking action a
            predicted_q = Q_s[:,a]

            # Calculate the loss
            loss = self.loss_function(target_q, predicted_q)
        
        # Backpropagate the gradient to the network's weights
        gradients = tape.gradient(loss, self.Qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.Qnet.trainable_variables))        

def train_Qnet(env, DQN, N_episodes=500, policy='egreedy', epsilon=0.2, temp=None):
    
    for episode in tqdm(range(N_episodes)):
        
        done = False

        # Initialize the start state
        s, info = env.reset(seed=42, return_info=True)
        
        while not done:
            # Select an action with the policy
            a = DQN.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

            # Simulate the environment
            s_next, r, done, info = env.step(a)

            # Update the Q network
            DQN.update(s, a, s_next, r, done)
            
            if done:
                break
            else:
                s = s_next # Update the state
            
            if episode >= N_episodes*0.8:
                env.render()

def test():
    # Create the environment
    env = gym.make("CartPole-v1")
    
    # Create the deep Q network
    net = DQN()
    
    train_Qnet(env, net)
    
if __name__ == '__main__':
    
    test()