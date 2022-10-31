from ray import rllib, tune
import ray
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
#from src.jss_lite.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
import gym
from src.jss_lite.jss_lite import jss_lite
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances
import time
from copy import deepcopy
import numpy as np
import os


def reload_ray_parameters(list):
    ray.shutdown()
    ModelCatalog.register_custom_model("dense_model", DenseModel)    
    def env_creator_variable_instance(config,env_name):
        return jssp_light_obs_wrapper_multi_instances([env_name])
    for instance_tune in list:
        tune.register_env('myEnv'+instance_tune[-8:-4], lambda config:env_creator_variable_instance(config=config,env_name=instance_tune))
        #print(f"{'myEnv'+instance_tune[-8:-4]} environment registered")
ray.shutdown()
curr_dir=(os.path.dirname(__file__))
#curr_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl'
instance_list_2=['/resources/jsp_instances/standard/la01.txt','/resources/jsp_instances/standard/la02.txt','/resources/jsp_instances/standard/la03.txt']
instance_list=['/resources/jsp_instances/standard/la01.txt']
#,'/resources/jsp_instances/standard/la04.txt','/resources/jsp_instances/standard/la05.txt'
instance_list_training=['/resources/jsp_instances/standard/la01.txt','/resources/jsp_instances/standard/la02.txt','/resources/jsp_instances/standard/la03.txt']
instance_list_validation=[]

# add current directory to path
instance_list_training=[curr_dir + s for s in instance_list_training]
instance_list=[curr_dir + s for s in instance_list]
instance_list_2=[curr_dir + s for s in instance_list_2]

# check if to create directories:
for check_instance in instance_list_training:
    instance_str=check_instance[-8:-4]
    s_path=(curr_dir+"/training_checkpoints/la_multi/"+instance_str)
    if not os.path.exists(s_path):
        os.mkdir(s_path)


train_agent=True
restore_agent= False
num_episodes = 10
config = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 4,
    "rollout_fragment_length": 50,
    "train_batch_size"  : 500,
    "sgd_minibatch_size": 64,
    #"lr"                : 0.0001,
    "lr"                : 0.001,
    "explore"           :True,
    #"horizon"           : 600,
    #"soft_horizon"      : True,
    "num_sgd_iter"      : 30,
    #"horizon"           : 100,
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
def env_creator(config):
    env=jssp_light_obs_wrapper_multi_instances(instances_list=instance_list)
    return env
reload_ray_parameters(instance_list_training)
tune.register_env('custom_jssp',env_creator)

# init checkpoint for untrained trainer:
agent = AlphaZeroTrainer( config=config, env='custom_jssp')
s_path=curr_dir+'/training_checkpoints/la_multi/untrained'
if not os.path.exists(s_path):
    os.mkdir(s_path)

# init untrained checkpoint to load config to
prev_checkpoint=agent.save_checkpoint(s_path)
print("start training")
# training loop:
for episode in range(num_episodes):
    for train_instance in instance_list_training:
        reload_ray_parameters(instance_list_training)
        # store string which instance is currently active
        instance_str=train_instance[-8:-4]
        agent = AlphaZeroTrainer( config=config, env='myEnv'+instance_str)
        agent.load_checkpoint(prev_checkpoint)
        t=time.time()
        agent.train()
        print(f"training iteration {episode} finished after {time.time()-t} seconds")
        s_path=(curr_dir+"/training_checkpoints/la_multi/"+instance_str+"/"+str(episode))
        if not os.path.exists(s_path):
            os.mkdir(s_path)
        prev_checkpoint=agent.save_checkpoint(s_path)
    print(f"{episode} of {num_episodes} finished")