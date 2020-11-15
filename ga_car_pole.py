
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
from ga import*


class GA_CAR_POLE(GA):
    def __init__(self,population_size,num_params,num_parents,num_mutations,range_low, range_high):
        super().__init__(population_size,num_params,num_parents,num_mutations,range_low, range_high) 
    
    def calculate_loss(self):
        loss = np.array([])
        # Optional: PPO2 requires a vectorized environment to run
        # the env is now wrapped automatically when passing it to the constructor
        # env = DummyVecEnv([lambda: env])
        # model = PPO2(MlpPolicy, env, verbose=1)

        # model.gamma = 0.99
        
        #loop through the population
        print(self.population)
        for i in range(self.population.shape[0]):
            env = gym.make('CartPole-v1')
            env.seed(0)
            model = PPO2(MlpPolicy, env, verbose=0)
            model.n_steps=128
            model.ent_coef = 0.01
            model.learning_rate = 0.00025
            model.vf_coef = self.population[i,:][0]
            model.max_grad_norm = self.population[i,:][1]
            model.lam = self.population[i,:][2]
            model.nminibatches = 4
            model.noptepochs = 4
            model.cliprange = 0.2
            model.gamma = self.population[i,:][3]
            model.seed = 0
            model.n_cpu_tf_sess = 1


            model.learn(total_timesteps=1000)

            obs = env.reset()
            total_reward = 0
            game_time = 500
            for i in range(game_time):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                total_reward += rewards
                # env.render()
                # time.sleep(.100)

            # print(total_reward)
            curr_loss = game_time - total_reward
            loss = np.append(loss, curr_loss)
            env.close()
        return loss