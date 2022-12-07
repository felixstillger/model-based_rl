from copy import deepcopy
import random
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
from src.jss_lite.jss_lite import jss_lite

# configs:
_max_jobs=10
_max_machines=10

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
   
class jssp_light_obs_wrapper_no_action_mask(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # shape multi:
        
        #shape_multi=(max(2*self.env.max_jobs,self.env.max_machines)*6)
        #shape_single=(max(2*self.env.n_jobs,self.env.n_machines)*6)
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space =gym.spaces.Box(low=0.0,high=1.0,
                    shape=(self.env.observation_space['obs'].shape),dtype=np.float64)
    
        
    def observation(self, obs):
        #print(obs)
        return obs['obs']
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
        return obs
    # additional unique:
    def get_legal_actions(self):
        return self.env.get_legal_actions()




class jssp_light_obs_wrapper_multi_instances(gym.Wrapper):
    def __init__(self, instances_list):
        self.instances_list=instances_list
        #super().__init__(env)
        instance=random.choice(self.instances_list)
        #print(f"{instance} is choosen as instance")
        self.env=jss_lite(instance_path=instance)
        # relevant parameters for wrapping:
        #just a parameter do define the max size of expected jobs
        self.max_jobs=_max_jobs
        self.max_machines=_max_machines
        self.dummy_jobs=self.max_jobs-self.env.n_jobs
        self.dummy_machines=self.max_machines-self.env.n_machines
        # differnce gives us the count of zeros to pad to the observation
        self.observation_padding_size=(max(self.max_jobs,self.max_machines)*7)-(max(self.env.n_jobs,self.env.n_machines)*7)
        self.action_mask_padding_size=2*self.dummy_jobs
        # gym declerations:
        self.action_space = gym.spaces.Discrete(2*self.max_jobs)
        self.observation_space = gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0,1,shape=(self.action_space.n,),dtype=np.int32),
                "obs": gym.spaces.Box(low=0.0,high=1.0,
                    shape=((max(self.max_jobs,self.max_machines)*7),),dtype=np.float64)
                     
                                                }
                                                )    
        # self.observation_space =gym.spaces.Dict(
        #     {
        #         'obs':self.env.observation_space['obs']+np.zeros(self.observation_padding_size),
        #         'action_mask':self.env.observation_space['action_mask']+np.zeros(self.action_mask_padding_size),
        #     }
        # )
    #def observation(self, obs):
        #print(obs)
    #    return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": np.asarray([*obs['action_mask'],*np.zeros(self.action_mask_padding_size,dtype=np.int32)])}
    def step(self,action):
        # check if action is out of bounds:
        trigger=False
        # if action==50:
        #     trigger=True
        # if trigger:
        #     print("1")
        # n_jobs=10
        # dummy_jobs=40

        if action >= self.env.n_jobs and action < self.env.n_jobs+self.dummy_jobs or action >= 2*self.env.n_jobs+self.dummy_jobs:
            # do nothing to the environment and get the old state:
            obs=self.env.observation_to_state()
            reward=0
            done=False
            info = {
            'action': action
        }
            #raise ValueError("this is if a not legal action is done")
            return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": self.transform_action_mask(obs['action_mask'])}, reward, done, info
        # check if dummy action is used and rescale it to no padding:
        if action>=self.env.n_jobs:
            if trigger:
                print("3")
            action=int(action-int(self.dummy_jobs))
        obs, reward, done, info = self.env.step(action)
        return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": self.transform_action_mask(obs['action_mask'])}, reward, done, info

    def reset(self):

        instance=random.choice(self.instances_list)
        #print(f"{instance} is choosen as instance")
        self.env=jss_lite(instance_path=instance, reward_mode='optimality gap')
        # relevant parameters for wrapping:
        #just a parameter do define the max size of expected jobs
        self.max_jobs=_max_jobs
        self.max_machines=_max_machines
        self.dummy_jobs=self.max_jobs-self.env.n_jobs
        self.dummy_machines=self.max_machines-self.env.n_machines
        # differnce gives us the count of zeros to pad to the observation
        self.observation_padding_size=(max(self.max_jobs,self.max_machines)*7)-(max(self.env.n_jobs,self.env.n_machines)*7)
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
        #return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": np.asarray([*obs['action_mask'],*np.zeros(self.action_mask_padding_size,dtype=np.int32)])}
        return {"obs": np.asarray([*obs['obs'],*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": self.transform_action_mask(obs['action_mask'])}

    def get_state(self):
        #return deepcopy(self.env),self.observation_padding_size,self.action_mask_padding_size
        return [deepcopy(self.env),self.observation_padding_size,self.action_mask_padding_size]
    
    def transform_action_mask(self,action_mask):
        # this definition is to split the action mask from real and dummy actions.
        #afterwards dummy actions are arranged at half mask size:
        #real_actions=action_mask[:self.env.n_jobs]
        #dummy_actions=action_mask[self.env.n_jobs:]
        #print(self.action_mask_padding_size)/2
        return np.asarray([*action_mask[:int(self.env.n_jobs)],*np.zeros(int(self.action_mask_padding_size/2),dtype=np.int32),*action_mask[int(self.env.n_jobs):],*np.zeros(int(self.action_mask_padding_size/2),dtype=np.int32)])

    
    def render(self,x_bar="Machine",y_bar="Job",start_count=0):
        self.env.render(x_bar=x_bar,y_bar=y_bar,start_count=start_count)

    def set_state(self, state):
        self.observation_padding_size=state[1]
        self.action_mask_padding_size=state[2]
        self.env = deepcopy(state[0])
        obs = np.ravel(self.env.observation)
        mask = self.env.get_legal_actions()
        return {"obs": np.asarray([*obs,*np.zeros(self.observation_padding_size,dtype=np.float64)]), "action_mask": self.transform_action_mask(mask)}

        # additional unique for ppo:
    def get_legal_actions(self):
        
        mask= self.env.get_legal_actions()
        return self.transform_action_mask(mask)

class Jssp_light_wrapper(gym.Env):
    """
    out of use
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

