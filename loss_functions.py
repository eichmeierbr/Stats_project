
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import tensorflow as tf
tf.set_random_seed(1234)
import time
from ga import*



def cartPoleLoss(params):
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    # model = PPO2(MlpPolicy, env, verbose=1)

    # model.gamma = 0.99
    
    #loop through the population
    env = gym.make('CartPole-v1')
    
    # env.seed(1)
    model = PPO2(MlpPolicy, env, verbose=0)
    model.n_steps=128
    model.ent_coef = 0.01
    model.learning_rate = params[0].value
    model.vf_coef = 0.5
    model.max_grad_norm = params[1].value
    model.lam = .2

    model.cliprange = 0.2
    model.gamma = .99
    # model.seed = 1
    # model.n_cpu_tf_sess = 1


    model.learn(total_timesteps=2500)

    obs = env.reset()
    total_reward = 0
    game_time = 500
    for j in range(game_time):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        if dones:
            break
        # env.render()
        # time.sleep(.100)

    env.close()
        
    
    return game_time-total_reward


# class TEST(GA):
#     def __init__(self,population_size,num_params,num_parents,num_mutations,range_low, range_high):
#         super().__init__(population_size,num_params,num_parents,num_mutations,range_low, range_high)

#     ##THIS function will interact will get loss from leaner
def test_loss(params):
    vals = []
    for p in params: vals.append(p.value)

    arr = np.array([1,10,20,30,40,50,60,70,80,90])
    val = np.array(vals)-arr
    loss = np.linalg.norm(val)
    return loss