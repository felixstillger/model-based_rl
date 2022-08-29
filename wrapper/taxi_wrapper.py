from logging import exception
import gym
from copy import deepcopy
import numpy as np
from copy import deepcopy

from pathlib import Path


"""
class TaxiTaxi(gym.Env):
    
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    

    def __init__(self, env_config):
        self.env = gym.make("Taxi-v3")
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.state=None
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return {
            "obs": self.env.reset(),
            "action_mask": np.array([1] * self.action_space.n, dtype=np.float32),
        }

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return (
            {"obs": obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)},
            score,
            done,
            info,
        )


    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        # obs = np.array(list(self.env.unwrapped.state))
        obs = self.env.observation(self.env.available_ops())
        state = self.encode(row, col, pass_idx, dest_idx)

        return {"obs": obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}

    def get_state(self):
        return deepcopy(self.env), self.running_reward
"""
class discretetobox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env, render_mode='human')
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = self.env.observation_space.n
        self.running_reward=0
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(0, 1, dtype=np.float32,shape=(self.n,)),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.initstate= None
        

    """
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (
            {"obs": obs, "action_mask": np.array([1, 1], dtype=np.float32)},
            rew,
            done,
            info,
        )
    """    
    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        return self.observation(self.s)
    
    def get_state(self):
        return deepcopy(self.env),self.running_reward

    def get_action_mask(self,obs):
        """"
    get action mask for valid actions
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

        Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    locs = {0:(0, 0), 1:(0, 4), 2:(4, 0), 3:(4, 3), 4:(-1,-1)}
    """
        # dictionary for locations
        locs = {0:(0, 0), 1:(0, 4), 2:(4, 0), 3:(4, 3)}

        mask=np.zeros(self.action_space.n,dtype=np.float32)
        taxi_row, taxi_col, pass_loc, dest_idx = (self.env.decode(obs))
        mask[0] = 1 if taxi_row != 5-1 else 0
        mask[1] = 1 if taxi_row != 0 else 0
        mask[2] = 1 if taxi_col != 5 -1 else 0
        mask[3] = 1 if taxi_col != 0 else 0
        mask[4] = 1 if pass_loc != 4 and (taxi_row,taxi_col) == locs[pass_loc] else 0
        mask[5] = 1 if pass_loc == 4 else 0


        #for exceptions or wrong programming:
        if taxi_row >=5:
            raise NotImplementedError
        return mask
    
    def observation(self, obs):
        # obs is int 
        new_obs = np.array(list(np.zeros(self.n,dtype=np.float32)))
        new_obs[obs]=1
        return {"obs": new_obs, "action_mask": self.get_action_mask(obs)}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return (
            self.observation(obs),
            score,
            done,
            info,
        )
    def reset(self):
        self.running_reward = 0
        obs=self.env.reset()
        return self.observation(obs)



class discretetobox2(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete), \
            "Should only be used to wrap Discrete envs."
        self.n = self.env.observation_space.n
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(0, 1, dtype=np.float32,shape=(self.n,)),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.action_space.n,)),
            }
        )
        self.initstate= None
        

    """
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (
            {"obs": obs, "action_mask": np.array([1, 1], dtype=np.float32)},
            rew,
            done,
            info,
        )
    """    
    def set_state(self, state):
        self.env = deepcopy(state)
        return self.observation(self.s)
    
    def get_state(self):
        return deepcopy(self.env)

    def get_action_mask(self,obs):
        """"
    get action mask for valid actions
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

        Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    locs = {0:(0, 0), 1:(0, 4), 2:(4, 0), 3:(4, 3), 4:(-1,-1)}
    """
        # dictionary for locations
        locs = {0:(0, 0), 1:(0, 4), 2:(4, 0), 3:(4, 3)}

        mask=np.zeros(self.action_space.n,dtype=np.float32)
        taxi_row, taxi_col, pass_loc, dest_idx = (self.env.decode(obs))
        mask[0] = 1 if taxi_row != 5-1 else 0
        mask[1] = 1 if taxi_row != 0 else 0
        mask[2] = 1 if taxi_col != 5 -1 else 0
        mask[3] = 1 if taxi_col != 0 else 0
        mask[4] = 1 if pass_loc != 4 and (taxi_row,taxi_col) == locs[pass_loc] else 0
        mask[5] = 1 if pass_loc == 4 else 0


        #for exceptions or wrong programming:
        if taxi_row >=5:
            raise NotImplementedError
        return mask
    
    def observation(self, obs):
        # obs is int 
        new_obs = np.array(list(np.zeros(self.n,dtype=np.float32)))
        new_obs[obs]=1
        return {"obs": new_obs, "action_mask": self.get_action_mask(obs)}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (
            self.observation(obs),
            rew,
            done,
            info,
        )
    def reset(self):
        obs=self.env.reset()
        return self.observation(obs)
    def render(self, render_mode):
        return self.env.render(mode="human")
        