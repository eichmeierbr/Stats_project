
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)

model.gamma = 0.99
model.n_steps=128
model.ent_coef = 0.01
model.learning_rate = 0.00025
model.vf_coef = 0.5
model.max_grad_norm = 0.5
model.lam = 0.95
model.nminibatches = 4
model.noptepochs = 4
model.cliprange = 0.2



model.learn(total_timesteps=10000)

obs = env.reset()
total_reward = 0
game_time = 500
for i in range(game_time):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    # env.render()
    # time.sleep(.100)

print(total_reward)
loss = game_time - total_reward
print(loss)
env.close()