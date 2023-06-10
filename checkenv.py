from stable_baselines3.common.env_checker import check_env
from obsenv import StayingAlive

env = StayingAlive()
env.reset()
print(env.canvas.shape)
print(env.reset())
print(env.observation_space)

check_env(env)