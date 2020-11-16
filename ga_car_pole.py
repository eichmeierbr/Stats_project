
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import time
import tensorflow as tf
from ga import*



class GA_CAR_POLE(GA):
    def __init__(self,population_size,num_params,num_parents,num_mutations,range_low, range_high):
        tf.set_random_seed(1234)
        super().__init__(population_size,num_params,num_parents,num_mutations,range_low, range_high) 
    
    def calculate_loss(self):
        loss = np.array([])
        # Optional: PPO2 requires a vectorized environment to run
        # the env is now wrapped automatically when passing it to the constructor
        # env = DummyVecEnv([lambda: env])
        # model = PPO2(MlpPolicy, env, verbose=1)

        # model.gamma = 0.99
        
        #loop through the population
        avg_num = 1
        # print(self.population)
        for i in range(self.population.shape[0]):
            curr_loss = 0
            for k in range(avg_num):
                env = gym.make('CartPole-v1')
                
                # env.seed(1)
                model = PPO2(MlpPolicy, env, verbose=0)
                model.n_steps=128
                model.ent_coef = 0.01
                model.learning_rate = self.population[i,:][0]
                model.vf_coef = 0.5
                model.max_grad_norm = self.population[i,:][1]
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
                    # env.render()
                    # time.sleep(.100)

                # print(-total_reward)
                curr_loss = curr_loss + game_time - total_reward
                env.close()
            loss = np.append(loss, curr_loss/avg_num)
            
        return loss


class TEST(GA):
    def __init__(self,population_size,num_params,num_parents,num_mutations,range_low, range_high):
        super().__init__(population_size,num_params,num_parents,num_mutations,range_low, range_high)

    ##THIS function will interact will get loss from leaner
    def calculate_loss(self):
        # this is for my simple tests, definitly needs to be improved
        arr = np.tile(np.array([1,10,20,30,40,50,60,70,80,90]), (self.population.shape[0], 1))
        val = self.population-arr
        loss = np.linalg.norm(val,axis=1)
        return loss