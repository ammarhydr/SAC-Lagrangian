import numpy as np
import torch
import safety_gym
import gym
import time
# import numpy as np
import matplotlib.pyplot as plt
from SAC_Agent import SACAgentWithCost
        

env_name = "Safexp-PointGoal2-v0"
max_steps = 1000
Training_Evaluation_Ratio = 4


env = gym.make(env_name)
n_actions = env.action_space.sample().size #safexp
state_size = env.reset().size

agent = SACAgentWithCost(n_actions, state_size)
scores_final=[]
cvs_final=[]
epochs=500

for i in range(epochs):
    evaluation_episode = i % Training_Evaluation_Ratio == 0
    print('\r', f'Episode: {i + 1}/{epochs}', end=' ')
    ep_step = 0
    score = 0
    cv = 0
    state=env.reset().reshape(1,-1)
    done=False
    while not done and ep_step < max_steps:
        ep_step += 1
        action = agent.act(state)
        next_observation, reward, done, info = env.step(action)
        next_state=next_observation.reshape(1,-1)
        if not evaluation_episode:
            start = time.time()
            agent.train_model(state, action, reward, next_state, info['cost'], done)
            end = time.time()
        else:
            score += reward
            if info['cost'] > 0:
                cv += 1
        state=next_state.copy()

        if done:
            break
    # print(ep_step)  
    if evaluation_episode:
        log_data = {"score":score, 'cv':cv}
        print(log_data)     
        scores_final.append(score)
        cvs_final.append(cv)
        
plt.plot(scores_final)
plt.title('State_value for Safety agent')
plt.show()
env.close()