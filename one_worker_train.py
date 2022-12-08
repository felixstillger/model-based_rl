# test to synchronize the workers:
from ray import rllib, tune
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
#from src.jss_lite.custom_torch_models import DenseModel_activation_relu as DenseModel
#from src.jss_lite.custom_torch_models import ConvNetModel
from ray.rllib.models.catalog import ModelCatalog
import gym
#from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer

from copy import deepcopy
import numpy as np
import pandas as pd
import os
import time
import ray
from src.jss_lite.jss_lite import jss_lite
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances


ray.shutdown()
ModelCatalog.register_custom_model("dense_model", DenseModel)

curr_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl'
curr_dir=(os.path.dirname(__file__))

num_inst=str(15)
instances_names='ima_'+num_inst+'_'+num_inst+'_no_act_10inner'
ima_inst_train=[]
ima_inst_test=[]

for i in range(0,20):
    ima_inst_train.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')
for i in range(21,30):
    ima_inst_test.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')


ray.shutdown()
ray.init()
ModelCatalog.register_custom_model("dense_model", DenseModel)    

def env_creator_random_instance(config_random):
    #env= jssp_light_obs_wrapper_multi_instances(instances_list=[loading_instance],env_config=config)
    env= jssp_light_obs_wrapper_multi_instances(instances_list=ima_inst_train,env_config=config_random)
    return env


tune.register_env('custom_jssp',lambda config: env_creator_random_instance(config))

config_eval = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 0,
    "rollout_fragment_length": 50,
    "train_batch_size"  : 500,
    "sgd_minibatch_size": 64,
    "lr"                : 0.0001,
    #"explore"           :False,
    #"horizon"           : 600,
    #"soft_horizon"      : True,
    "num_sgd_iter"      : 1,
    #"horizon"           : 100,
    "mcts_config"       : {
        "puct_coefficient"   : 1.5,
        "num_simulations"    : 200,
        "temperature"        : 1,
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

s_path=(curr_dir+'/training_checkpoints'+"/"+'oneworker'+num_inst)
if not os.path.exists(s_path):
    os.mkdir(s_path)


#s_path='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/alpha_zero_random_instances'
agent = AlphaZeroTrainer( config=config_eval, env='custom_jssp')
for _ in range(2000):
    agent.train()
    agent.save_checkpoint(s_path)