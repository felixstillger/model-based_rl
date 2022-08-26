from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
import src.jss_graph_env.disjunctive_graph_jss_env as jss_env
import src.jsp_instance_parser 

class jssp_obs_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": self.env.action_space,
            }
        )
    def observation(self, obs):
        # obs is int 
        print(obs)
        return {"obs": obs['observation'], "action_mask": obs['action_mask']}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
        



class Jssp_wrapper(gym.Env):
    """
    Wrapper for the custom jssp Problem.

    clipboard:
    path='resources/jsp_instances/standard/abz8.txt'
    curr_instance=src.jsp_instance_parser.parse_jps_standard_specification(path)
    res,std_matrix=curr_instance
    env = 

    """
    """
    def __init__(self, config=None,visualisation='gantt_console',instance_path='resources/jsp_instances/standard/abz8.txt'):
        res,std_matrix=src.jsp_instance_parser.parse_jps_standard_specification(instance_path)
        self.env=jss_env.DisjunctiveGraphJssEnv(res,default_visualisations=visualisation)
        self.action_space = gym.spaces.Discrete(self.env.total_tasks_without_dummies)
        self.observation_space = Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),
            }
        )
        self.observation_space = self.env.observation_space
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return self.env.reset()
            

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (
            {"obs": obs, "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32)},
            rew,
            done,
            info,
        )
        return self.env.step(action)

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))


        return {"obs": obs, "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32)}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
    """
    #def render(self):
    #    self.env.render()

