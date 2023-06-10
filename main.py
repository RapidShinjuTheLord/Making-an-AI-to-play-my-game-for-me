#1: Import Dependencies
import os
import sys
import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#2: Import Env
#from envclass import StayingAlive
#from arrayobsenv import StayingAlive
from fiveplasma import StayingAlive
env = StayingAlive(True)
#3: envtesting
episodes = 2
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        env.clock.tick(160)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward

    #    print([(item.pos) for item in env.board.obstacles])
    #print('Episode:{} Score:{}'.format(episode, score))
env.close()
#4: trainagentmodel
log_path = os.path.join('Agent_Directory', 'logs')
model = PPO('MlpPolicy', env, verbose=1)
#model = DQN('MlpPolicy', env, verbose=1)
#model = A2C('MlpPolicy', env, verbose=1)
#pre-5: customize agent training
#custom_objects = {'learning_rate' : 0.003}
custom_objects = {'learning_rate' : 0.001}
model = PPO.load('Agent_Directory/savedmodels/untrained.zip', env, custom_objects = custom_objects)
#model = PPO.load("dir...", env=env, custom_objects = custom_objects)
#8
custom_objects = {'learning_rate' : 0.0005}
model = PPO.load('Agent_Directory/savedmodels/' + NAME, env, custom_objects = custom_objects)
#5: train
model.learn(total_timesteps=10000000, reset_num_timesteps=False)
#5b
TIMESTEPS = 10000
for i in range(1, 10 + 1):
    model.learn(total_timesteps = TIMESTEPS, reset_num_timesteps=False)
    #print(model.env.item.pos)
    model.save(f'Agent_Directory/savedmodels/100kPPO7layerv2/{TIMESTEPS*i}')
#6: test
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        env.clock.tick(160)
#7:
NAME = 'agent_in_full_environment_PPO_boundary_long_training'
model.save("Agent_Directory/savedmodels/" + NAME)

