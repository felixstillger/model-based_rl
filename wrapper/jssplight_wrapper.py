from copy import deepcopy
import random
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
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=start_count)

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.ravel(self.env.observation)
        mask = self.env.get_legal_actions()
        return {"obs":obs,"action_mask":mask}
   




class jssp_light_obs_wrapper_multi_instances(gym.Wrapper):
    def __init__(self, instances_list):
        self.instances_list=instances_list
        #super().__init__(env)
        instance=random.choice(self.instances_list)
        print(f"{instance} is choosen as instance")
        self.env=jss_lite(instance_path=instance)
        # relevant parameters for wrapping:
        #just a parameter do define the max size of expected jobs
        self.max_jobs=50
        self.max_machines=50
        self.dummy_jobs=self.max_jobs-self.env.n_jobs
        self.dummy_machines=self.max_machines-self.env.n_machines
        # differnce gives us the count of zeros to pad to the observation
        self.observation_padding_size=(max(2*self.max_jobs,self.max_machines)*6)-(max(2*self.env.n_jobs,self.env.n_machines)*6)
        self.action_mask_padding_size=2*self.dummy_jobs
        # gym declerations:
        self.action_space = gym.spaces.Discrete(2*self.max_jobs)
        self.observation_space = gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0,1,shape=(self.action_space.n,),dtype=np.int32),
                "obs": gym.spaces.Box(low=0.0,high=1.0,
                    shape=((max(2*self.max_jobs,self.max_machines)*6),),dtype=np.float64)
                     
                                                }
                                                )    
        # self.observation_space =gym.spaces.Dict(
        #     {
        #         'obs':self.env.observation_space['obs']+np.zeros(self.observation_padding_size),
        #         'action_mask':self.env.observation_space['action_mask']+np.zeros(self.action_mask_padding_size),
        #     }
        # )
    def observation(self, obs):
        #print(obs)
        return {"obs": obs['obs']+np.zeros(self.observation_padding_size), "action_mask": obs['action_mask']+np.zeros(self.action_mask_padding_size)}

    def reset(self):

        instance=random.choice(self.instances_list)
        print(f"{instance} is choosen as instance")
        self.env=jss_lite(instance_path=instance)
        # relevant parameters for wrapping:
        #just a parameter do define the max size of expected jobs
        self.max_jobs=50
        self.max_machines=50
        self.dummy_jobs=self.max_jobs-self.env.n_jobs
        self.dummy_machines=self.max_machines-self.env.n_machines
        # differnce gives us the count of zeros to pad to the observation
        self.observation_padding_size=(max(2*self.max_jobs,self.max_machines)*6)-(max(2*self.env.n_jobs,self.env.n_machines)*6)
        self.action_mask_padding_size=2*self.dummy_jobs
        # gym declerations:
        # self.action_space = gym.spaces.Discrete(2*self.max_jobs)    
        # self.observation_space =gym.spaces.Dict(
        #     {
        #         'obs':self.env.observation_space['obs']+np.zeros(self.observation_padding_size),
        #         'action_mask':self.env.observation_space['action_mask']+np.zeros(self.action_mask_padding_size),
        #     }
        # )
        obs=self.env.reset()
        return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": np.asarray([*obs['action_mask'],*np.zeros(self.action_mask_padding_size,dtype=np.int32)])}

    def get_state(self):
        return deepcopy(self.env),self.observation_padding_size,self.action_mask_padding_size

    
    def render(self,x_bar="Machine",y_bar="Job",start_count=0):
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=start_count)

    def set_state(self, state,obs_padding,action_mask_padding):
        self.observation_padding_size=obs_padding
        self.action_mask_padding_size=action_mask_padding
        self.env = deepcopy(state)
        obs = np.ravel(self.env.observation)+np.zeros(self.observation_padding_size)
        mask = self.env.get_legal_actions()+np.zeros(self.action_mask_padding_size)
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

