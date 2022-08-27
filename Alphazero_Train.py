# %%
import gym
from wrapper.taxi_wrapper import discretetobox
# %%
from ray import rllib, tune
# depending on ray version use : or no contrib
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
#from ray.rllib.algorithms.alpha_zero import AlphaZeroConfig
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
#from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole

from ray.rllib.models.catalog import ModelCatalog
from wrapper.jssp_wrapper import Jssp_wrapper
from wrapper.taxi_wrapper import discretetobox
#from wrapper.taxi_wrapper import TaxiTaxi
ModelCatalog.register_custom_model("dense_model", DenseModel)
#register_env("CartPoleEnv", lambda _: CartPole())
register_env("JsspEnv", lambda _: Jssp_wrapper())
register_env("Taxi-v3", lambda _:discretetobox())
#register_env("Taxi-v3", lambda _:TaxiTaxi())

from copy import deepcopy
import numpy as np

checkpoint_path=f'/home/fs608798/masterarbeit/model-based_rl/training_checkpoints/checkpoints_az_taxi/checkpoint-10'
load_from_checkpoint=True

config = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 4,
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
    return env

# use tune to register the custom environment for the ppo trainer
tune.register_env('TaxiTaxi',env_creator)

import time

print("init agent:")
agent = AlphaZeroTrainer( config=config, env='TaxiTaxi')

print("agent initialized")

# use string number to restore
# nr_restore="10"
if load_from_checkpoint:
	agent.load_checkpoint(checkpoint_path)
	print("agent loaded from checkpoint")
#print("awd")
#agent.restore("checkpoints_az/rllib_checkpoint1/checkpoint_000001/checkpoint-1")
#agent.load_checkpoint("checkpoints_az/checkpoint-44")
print("start training")
for _ in range(0,150):
    tmp_time=time.time()
    agent.train()
    print(f"training iteration {_} finished after {time.time() - tmp_time} seconds")
    #agent.save(f"save_az/rllib_checkpoint{_}")
    agent.save_checkpoint(f"training_checkpoints/checkpoints_az_taxi_overnight")
    #agent.save_to_object(f"objects_az/rllib_checkpoint{_}")
   

# %%
agent = AlphaZeroTrainer( config=config,env=CartPole)
agent.load_checkpoint("checkpoints_az_2/checkpoint-148")

import time
policy = agent.get_policy(DEFAULT_POLICY_ID)




