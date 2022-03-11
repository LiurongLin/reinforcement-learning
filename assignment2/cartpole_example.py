import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    env.render(mode='human')
    
    # s = env.state()
    # a = act_f(s) (action function is built with Q as argument)
    # s', r, done = env.step(a)
    # td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
    # 
    action = env.action_space.sample()
    
    observation, reward, done, info = env.step(action)
    print(observation, reward, info)
    

    if done:
        observation, info = env.reset(return_info=True)
env.close()

print('observation:', observation)
print('reward:', reward)
print('done:', done)
print('action:', action)
