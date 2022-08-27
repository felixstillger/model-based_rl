# %%
import gym
from wrapper.taxi_wrapper import discretetobox
env=discretetobox(gym.make('Taxi-v3'))
state=env.reset()

taxi_row, taxi_col, pass_loc, dest_idx = env.decode(env.s)
print(taxi_row)
print(taxi_col)
print(pass_loc)
print(env.locs[pass_loc])
print((taxi_row,taxi_col))
print(env.locs[pass_loc])
print(dest_idx)
env.render()
print(state['action_mask'])
"""
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
"""
print(True) if (taxi_row,taxi_col) == env.locs[pass_loc] else print(False)

# %%
from ray import rllib, tune
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
from wrapper.jssp_wrapper import Jssp_wrapper
import gym
from wrapper.taxi_wrapper import discretetobox
#from wrapper.taxi_wrapper import TaxiTaxi
ModelCatalog.register_custom_model("dense_model", DenseModel)
register_env("CartPoleEnv", lambda _: CartPole())
register_env("JsspEnv", lambda _: Jssp_wrapper())
register_env("Taxi-v3", lambda _:discretetobox())
#register_env("Taxi-v3", lambda _:TaxiTaxi())

from copy import deepcopy
import numpy as np

        


config = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 6,
    "rollout_fragment_length": 50,
    "train_batch_size"  : 500,
    "sgd_minibatch_size": 64,
    "lr"                : 0.0001,
    "num_sgd_iter"      : 1,
    "mcts_config"       : {
        "puct_coefficient"   : 1.5,
        "num_simulations"    : 100,
        "temperature"        : 1.0,
        "dirichlet_epsilon"  : 0.20,
        "dirichlet_noise"    : 0.03,
        "argmax_tree_policy" : False,
        "add_dirichlet_noise": True,
    },
    "ranked_rewards"    : {
        "enable": True,
    },
    "model"             : {
        "custom_model": "dense_model",

    },
}

def env_creator(env_config):
    env = discretetobox(gym.make("Taxi-v3"))
    #env = gym.make('LunarLander-v2')
    return env

# use tune to register the custom environment for the ppo trainer
tune.register_env('TaxiTaxi',env_creator)

#env = discretetobox(gym.make("Taxi-v3"))
#env2 = CartPole()

#print(env2.observation_space)

#print(env2.env.observation_space)



#print(env.observation_space['obs'])

#print(env.env.observation_space)


#tmp=(env.observation_space['action_mask'])
#tmp2=(env2.observation_space['action_mask'])


# %%
# checkpoint_path = analysis.get_last_checkpoint() or args.checkpoint
import time
#agent = AlphaZeroTrainer(env=Jssp_wrapper, config=config)
#agent = AlphaZeroTrainer( config=config,env=CartPole)
agent = AlphaZeroTrainer( config=config, env='TaxiTaxi')
#agent = AlphaZeroTrainer( config=config, env=TaxiTaxi)
# use string number to restore
# nr_restore="10"
#checkpoint_path=f'checkpoints_az/rllib_checkpoint{nr_restore}/checkpoint_{nr_restore.zfill(6)}/checkpoint-{nr_restore}'
#agent.load_checkpoint("checkpoints_az/rllib_checkpoint1")
#print("awd")
#agent.restore("checkpoints_az/rllib_checkpoint1/checkpoint_000001/checkpoint-1")
#agent.load_checkpoint("checkpoints_az/checkpoint-44")
print("start training")
for _ in range(0,10):
    tmp_time=time.time()
    agent.train()
    print(f"training iteration {_} finished after {time.time() - tmp_time} seconds")
    #agent.save(f"save_az/rllib_checkpoint{_}")
    agent.save_checkpoint(f"training_checkpoints/checkpoints_az_taxi")
    #agent.save_to_object(f"objects_az/rllib_checkpoint{_}")
    


# %%
# ray 1.13.0 ezpip uninstall -y ray
#agent = AlphaZeroTrainer( config=config,env=CartPole)
#agent.load_checkpoint("checkpoints_az/checkpoint-0")


# %%
agent = AlphaZeroTrainer( config=config,env=CartPole)
agent.load_checkpoint("checkpoints_az_2/checkpoint-148")

import time
policy = agent.get_policy(DEFAULT_POLICY_ID)

env = CartPole()
#env = Jssp_wrapper()
obs = env.reset()

episode = MultiAgentEpisode(
    PolicyMap(0,0),
    lambda _, __: DEFAULT_POLICY_ID,
    lambda: None,
    lambda _: None,
    0,
)

episode.user_data['initial_state'] = env.get_state()


action, _, _ = policy.compute_single_action(obs, episode=episode)
print(action)
obs, reward, done, _ = env.step(action)


done = True

while not done:
    action, _, _ = policy.compute_single_action(obs, episode=episode)
    print(action)
    obs, reward, done, _ = env.step(action)
    print(obs)
    env.render()
    #time.sleep(0.1)
    episode.length += 1

assert reward == episode.length
env.close()

# %%


# %%



