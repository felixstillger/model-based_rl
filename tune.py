from ray import rllib, tune
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
#from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from src.jss_lite.custom_torch_models import DenseModel_activation_relu as DenseModel
#from src.jss_lite.custom_torch_models import ConvNetModel
from ray.rllib.models.catalog import ModelCatalog
import gym
#from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer

from src.jss_lite.jss_lite import jss_lite
ModelCatalog.register_custom_model("dense_model", DenseModel)
from copy import deepcopy
import numpy as np



import os
curr_dir=(os.path.dirname(__file__))

instance_list=['/resources/jsp_instances/standard/la01.txt','/resources/jsp_instances/standard/la02.txt','/resources/jsp_instances/standard/la03.txt','/resources/jsp_instances/standard/la04.txt','/resources/jsp_instances/standard/la05.txt']
instance_list=[curr_dir + s for s in instance_list]
instance_path=curr_dir+'/resources/jsp_instances/standard/ft06.txt'
checkpoint_path='/training_checkpoints/checkpoints_tune'

from wrapper.jssplight_wrapper import jssp_light_obs_wrapper
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances

def env_creator(config):
    env = jssp_light_obs_wrapper(jss_lite(instance_path=instance_path))
    #env=jssp_light_obs_wrapper_multi_instances(instances_list=instance_list)
    return env


# use tune to register the custom environment for the ppo trainer
tune.register_env('custom_jssp',env_creator)

tune.run(
    "contrib/AlphaZero",
    stop={"training_iteration": 500},
    local_dir="training_checkpoints/checkpoints_tune",
    max_failures=0,
    checkpoint_freq = 1,
    config={
        "env": 'custom_jssp',
        "disable_env_checking":True,
        "num_workers": 4,
        "rollout_fragment_length": 50,
        "train_batch_size": 500,
        "sgd_minibatch_size": 32,
        "lr": 1e-4,
        "horizon": 1000,
        "num_sgd_iter": 1,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 100,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": False,
        },
        "ranked_rewards": {
            "enable": True,
        },
        "model": {
            "custom_model": "dense_model",
        },
        "evaluation_interval": 0,
        "evaluation_config": {
            "render_env": True,
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
            },
        },
    },
)
