# load agent on environment:
from copy import deepcopy
import os
import pandas as pd
from ray import rllib, tune
import ray
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
#from src.jss_lite.custom_torch_models import DenseModel_activation_relu as DenseModel

#from src.jss_lite.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
import gym
from src.jss_lite.jss_lite import jss_lite
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances
def get_instance_name(string):
    return string.replace('/', ' ').split(' ')[-1].split('.')[-2]

curr_dir=(os.path.dirname(__file__))



checkpoint_path='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/one_environment/oneenvironment6/checkpoint-'
loading_instance="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_4_inst.json"
#train_instances=[loading_instance]
train_instances=[]
num_checkpoints=10
num_inst=str(6)
for i in range(4,7):
    train_instances.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')

#train_instances=["/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_"+str(i)+"_inst.json" for i in range(20)]
#checkpoint_path="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/ima_15_15_no_act_10inner/15x15_2_inst/0/checkpoint-5"
results={}



for loading_instance in train_instances:
    every_action_time=[]
    train_rewards=[]
    every_action=[]
    ray.shutdown()
    ray.init()
    ModelCatalog.register_custom_model("dense_model", DenseModel)    
    def env_creator(config):
        env= jssp_light_obs_wrapper_multi_instances([loading_instance])
        return env

    tune.register_env('custom_jssp',env_creator)

    config_eval = {
        "framework": "torch",
        "disable_env_checking":True,
        "num_workers"       : 0,
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
            "num_simulations"    : 5,
            "temperature"        : 1,
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
    agent = AlphaZeroTrainer( config=config_eval, env='custom_jssp')
    for iteration in range(1,num_checkpoints+1):
        agent.load_checkpoint(checkpoint_path+str(iteration))
        import time
        length_list=[]
        reward_list=[]
        time_list=[]
        for _ in range(1):
            policy = agent.get_policy(DEFAULT_POLICY_ID)
            action_list=[]
            env = env_creator("config")
            obs = env.reset()
            episode = MultiAgentEpisode(
                PolicyMap(0,0),
                lambda _, __: DEFAULT_POLICY_ID,
                lambda: None,
                lambda _: None,
                0,
            )
            episode.user_data['initial_state'] = env.get_state()
            done = False
            while not done:
                begin_time=time.time()
                action, _, _ = policy.compute_single_action(obs, episode=episode)
                action_list.append(action)
                time_list.append(time.time()-begin_time)
                obs, reward, done, _ = env.step(action)
                episode.length += 1
            print(reward)
            #env.render()
        every_action_time.append(time_list)
        train_rewards.append(reward)
        every_action.append(action_list)
    
    results[get_instance_name(loading_instance)]={'train_reward':train_rewards,'every_action_time':every_action_time,'every_action':every_action}
    df=pd.DataFrame.from_dict(results)
    df.to_csv(f"eval_{num_inst}_inst_{num_checkpoints}_check.csv")   