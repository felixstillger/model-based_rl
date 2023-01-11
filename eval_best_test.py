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
def multi_env_decoder(iter,size):
        mod=iter % 3
        mod_2=floor(iter/60)
        mod_3=floor(iter/3)
        if size=='15':
            mod=1
            mod_2=0
            mod_3=size
            return f"/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/merged_multi_env/ima_{size}_{size}_1_inner_500sim/{size}x{size}_{mod_3}_inst/{mod_2}/checkpoint-{str(mod)}"

        else:
            return f"/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/merged_multi_env/ima_{size}_{size}_1_inner_500sim/{size}x{size}_{mod_3}_inst/{mod_2}/checkpoint-{str(mod+1)}"
from math import floor
ids=['500_sims_multi_env', 'one_env_100', 'one_env_2000']
best={'6':['31', '32', '10'],'8': ['1', '69', '5'],'10': ['1', '47', '4'],'15': ['1', '55', '1']}

def main():
    #loading_instance="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_4_inst.json"
    #train_instances=[loading_instance]
    train_instances=[]
    test_instances=[]

    # here num checkpoints act as maximal number of loaded checkpoints
    num_checkpoints=3
    num_inst=str(6)
    tmp_inst=[15]
    for num_inst in tmp_inst:
        num_inst=str(num_inst)
        checkpoints_p=[]
        i=best[num_inst][0]
        checkpoints_p.append([multi_env_decoder(int(i),num_inst)])
        i=best[num_inst][1]
        checkpoints_p.append(['/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/one_environment/oneenvironment'+num_inst+'/checkpoint-'+str(i)])
        i=best[num_inst][2]
        checkpoints_p.append(['/Users/felix/sciebo/masterarbeit/progra/model-based_rl/cluster_daten/one_environment_2000/oneenvironmentima_'+num_inst+'_'+num_inst+'_2000_sims/checkpoint-'+str(i)])
        #print(checkpoints_p[0])
        train_instances=[]
        test_instances=[]
        for i in range(4,6):
            train_instances.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')
        for i in range(20,21):
            test_instances.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')
        #train_instances=["/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/15x15x15/15x15_"+str(i)+"_inst.json" for i in range(20)]
        #checkpoint_path="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/ima_15_15_no_act_10inner/15x15_2_inst/0/checkpoint-5"
        results={}

        for loading_instance in test_instances:#train_instances:
            every_action_time=[]
            train_rewards=[]
            every_action=[]
        #    for sim in num_sims:
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
            for iteration in range(0,num_checkpoints):
                checkpoint_list=checkpoints_p[iteration]
                for c_nr,checkpoint_path in enumerate(checkpoint_list):
                    c_nr+=1    
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
                        print(reward)
                        #env.render()
                    every_action_time.append(time_list)
                    train_rewards.append(reward)
                    every_action.append(action_list)
                    # write results
                    results[get_instance_name(loading_instance)+"_"+ids[iteration]+"_"+str(best[num_inst][iteration])]={'train_reward':reward,'every_action_time':time_list,'every_action':action_list,'checkpoint nr':c_nr}
                    #results[sim]={'train_reward':train_rewards,'every_action_time':every_action_time,'every_action':every_action}

                #df=pd.DataFrame.from_dict(results)
                #df.to_csv(f"best_checkpoints_tests_size{num_inst}_inst_20_220_2sims_1_5puct.csv")   
                    #df.to_csv(f"different_num_sims_az_1_5_puct_inst4.csv")  

if __name__ == "__main__":
    main()
