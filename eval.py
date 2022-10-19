from tracemalloc import start
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
import pandas as pd
import os
import time

curr_dir=(os.path.dirname(__file__))

instance_list=['/resources/jsp_instances/standard/la01.txt','/resources/jsp_instances/standard/la02.txt','/resources/jsp_instances/standard/la03.txt','/resources/jsp_instances/standard/la04.txt','/resources/jsp_instances/standard/la05.txt']
instance_list=[curr_dir + s for s in instance_list]
instance_path=curr_dir+'/resources/jsp_instances/standard/ft06.txt'
checkpoint_path='/training_checkpoints/checkpoints_tune'

from wrapper.jssplight_wrapper import jssp_light_obs_wrapper
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances

def env_creator(config):
    #env = jssp_light_obs_wrapper(jss_lite(instance_path=instance_path))
    env=jssp_light_obs_wrapper_multi_instances(instances_list=instance_list)
    return env
def env_creator_single(instance):
    env=jssp_light_obs_wrapper_multi_instances(instances_list=instance)
    return env

def eval_env(agent,env):
    state=env.reset()
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    episode = MultiAgentEpisode(
        PolicyMap(0,0),
        lambda _, __: DEFAULT_POLICY_ID,
        lambda: None,
        lambda _: None,
        0,
    )
    episode.user_data['initial_state'] = env.get_state()
    done = False
    episode_length=0
    start_time=time.time()
    while not done:
        action, _, _ = policy.compute_single_action(state, episode=episode)
        state, reward, done, _ = env.step(action)
        episode.length+=1
        episode_length += 1
    return time.time()-start_time, episode_length, reward

# use tune to register the custom environment for the ppo trainer
tune.register_env('custom_jssp',env_creator)
#tune.register_env('custom_jssp_single',env_creator())
config = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 6,
    "rollout_fragment_length": 50,
    "train_batch_size"  : 500,
    "sgd_minibatch_size": 64,
    "lr"                : 0.0001,
    "explore"           :False,
    #"horizon"           : 600,
    #"soft_horizon"      : True,
    "num_sgd_iter"      : 1,
    #"horizon"           : 100,
    "mcts_config"       : {
        "puct_coefficient"   : 1.5,
        "num_simulations"    : 100,
        "temperature"        : 1.5,
        "dirichlet_epsilon"  : 0.20,
        "dirichlet_noise"    : 0.03,
        "argmax_tree_policy" : True,
        "add_dirichlet_noise": False,
    },
    "ranked_rewards"    : {
        "enable": True,
    },
    "model"             : {
        "custom_model": "dense_model",

    },
}



agent = AlphaZeroTrainer( config=config, env='custom_jssp')
#if restore_agent:
#restore_path='training_checkpoints/checkpoints_az_jsslite/checkpoint-5'
training_folder='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/published_checkpoints/contrib_AlphaZero_custom_jssp_50160_00000_0_2022-10-17_18-46-57'
#training_folder='/home/fs608798/masterarbeit/model-based_rl/training_checkpoints/checkpoints_tune/contrib/AlphaZero/contrib_AlphaZero_custom_jssp_50160_00000_0_2022-10-17_18-46-57'
nr_checkpoints=0
for f in os.listdir(training_folder):
    if 'checkpoint' in f:
        nr_checkpoints+=1
eval_sheet={}
run=0
for f in os.listdir(training_folder):
    if 'checkpoint' in f:
        restore_path=training_folder+'/'+f+'/checkpoint-'+str(int(f[-6:]))
        if os.path.exists(restore_path):
            agent.load_checkpoint(restore_path)
            # here comes the evaluation:
            for instance in instance_list:
                eval_sheet[str(f[-6:]),instance]=eval_env(agent,jssp_light_obs_wrapper_multi_instances(instances_list=[instance]))
            run+=1
            print(f"run: {run} of {nr_checkpoints} evaluated")
            df=pd.DataFrame.from_dict(eval_sheet)
            df.to_csv('eval.csv')

# generate csv    
df=pd.DataFrame.from_dict(eval_sheet)
df.to_csv('eval.csv')



        


