from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
from src.jss_lite.jss_lite import jss_lite

class jssp_light_obs_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space =gym.spaces.Dict(
            {
                'obs':self.env.observation_space['obs'],
                'action_mask':self.env.observation_space['action_mask'],
            }
        )
        
    def observation(self, obs):
        #print(obs)
        return {"obs": obs['obs'], "action_mask": obs['action_mask']}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
    def get_state(self):
        return deepcopy(self.env)

    
    def render(self,x_bar="Machine",y_bar="Job",start_count=0):
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=0)

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.ravel(self.env.observation)
        mask = self.env.get_legal_actions("g")
        return {"obs":obs,"action_mask":mask}
   

class jssp_light_obs_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space =gym.spaces.Dict(
            {
                'obs':self.env.observation_space['obs'],
                'action_mask':self.env.observation_space['action_mask'],
            }
        )
        
    def observation(self, obs):
        #print(obs)
        return {"obs": obs['obs'], "action_mask": obs['action_mask']}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
    def get_state(self):
        return deepcopy(self.env)

    
    def render(self,x_bar="Machine",y_bar="Job",start_count=0):
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=0)

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.ravel(self.env.observation)
        mask = self.env.get_legal_actions("g")
        return {"obs":obs,"action_mask":mask}







class Jssp_light_wrapper(gym.Env):
    """
    Wrapper for the custom jssp Problem.

    clipboard:
    path='resources/jsp_instances/standard/abz8.txt'
    curr_instance=src.jsp_instance_parser.parse_jps_standard_specification(path)
    res,std_matrix=curr_instance
    env = 

    """
    
    def __init__(self, config=None):
        self.env = jss_lite(instance_path='resources/jsp_instances/standard/ft06.txt')
        self.action_space = Discrete(self.env.action_space.n)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space['obs'],
                "action_mask": self.env.observation_space['action_mask'],
            }
        )

    def reset(self):
        return self.env.reset()
            

    def step(self, action):
        # obs, rew, done, info = self.env.step(action)
        # return (
        #     obs,
        #     rew,
        #     done,
        #     info,
        # )
        return self.env.step(action)

    def observation(self,obs):
        r_obs=obs['obs']
        r_a_mask=obs['action_mask']
        return(
            {"obs":r_obs,"action_mask":r_a_mask,
            }
        )

    def set_state(self, state):
        self.env = deepcopy(state)
        #obs = np.array(list(self.env.unwrapped.state))
        return {"obs": self.env.observation_space['obs'], "action_mask": self.env.observation_space['action_mask']}

    def get_state(self):
        return deepcopy(self.env)
    
    def render(self,x_bar="Machine",y_bar="Job",start_count=0):
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=0)

