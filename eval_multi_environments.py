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
def main():
    def get_instance_name(string):
        return string.replace('/', ' ').split(' ')[-1].split('.')[-2]

    curr_dir=(os.path.dirname(__file__))



    #loading_instance="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_4_inst.json"
    #train_instances=[loading_instance]
    train_instances=[]
    # here equal to inner episodes
    num_checkpoints=1
    inst_l=[3,6,8,10,15]
    num_sims=[2,5,20,50]
    num_inst=str(15)
    counter=-1
    num_train_inst=20
    results={}
    out_episodes=1
    checkpoint_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/merged_multi_env/ima_'+num_inst+'_'+num_inst+'_1_inner_500sim/'
    #checkpoint_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/ima_'+num_inst+'_'+num_inst+'_no_act_10inner/'
    train_instances=[]
    for i in range(0,20):
        train_instances.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')

    #train_instances=["/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_"+str(i)+"_inst.json" for i in range(20)]
    #checkpoint_path="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/ima_15_15_no_act_10inner/15x15_2_inst/0/checkpoint-5"



    print('start loading')
    for loading_instance in train_instances:
        counter+=1
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
                "num_simulations"    : 2,
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
        for outer in range(out_episodes):

            for iteration in range(1,num_checkpoints+1):
                checkpoint_path=checkpoint_dir+get_instance_name(loading_instance)+"/"+str(outer)+"/checkpoint-"+str(iteration)
                if os.path.exists(checkpoint_path):
                    agent.load_checkpoint(checkpoint_path)
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
                        #print(reward)
                        #env.render()
                    every_action_time.append(time_list)
                    train_rewards.append(reward)
                    every_action.append(action_list)
                    # write results
                    #results[get_instance_name(loading_instance),outer,iteration]={'train_reward':reward,'every_action_time':sum(time_list),'every_action':action_list,'outer':outer,'iteration':iteration,'order':outer*60+counter*3+iteration}
                    results[str(outer*60+counter*num_checkpoints+iteration)]={'instance':get_instance_name(loading_instance), 'train_reward':reward,'every_action_time':sum(time_list),'every_action':action_list,'outer':outer,'iteration':iteration,'order':outer*60+counter*3+iteration}

                    df=pd.DataFrame.from_dict(results)
                    df.to_csv(f"eval_merged_multi_size{num_inst}_500sims_v2.csv")
                    print(outer*60+counter*num_checkpoints+iteration)
                    
                else:
                    print(checkpoint_path)

  

if __name__ == "__main__":
    main()