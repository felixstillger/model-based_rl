from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import ray
import gym
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_no_action_mask
from src.jss_lite.jss_lite import jss_lite
from ray import tune
import numpy as np
import pandas as pd

curr_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl'
inst='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/8x8x8/8x8_5_inst.json'#curr_dir + '/resources/jsp_instances/standard/la01.txt'
# Configure the algorithm.
def env_creator(env_config):
    env=jssp_light_obs_wrapper_no_action_mask(jssp_light_obs_wrapper_multi_instances(instances_list=[inst]))
    return env
from wrapper.jssp_wrapper_klagenfurt import jssp_klagenfurt_obs_wrapper
import JSSEnv
ray.shutdown()
tune.register_env('custom_jssp',env_creator)
config = {
    "env": "custom_jssp",
    "disable_env_checking":True,
    "num_workers": 4,
    "framework": "tf",
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "relu",
    },
    "evaluation_num_workers": 0,
    "evaluation_config": {
        "render_env": False,
    },
    
}
trainer = PPOTrainer(config=config)
sizes=[3,6,8,10,15]
check_dic_max={'3':500,'6':343,'8':162,'10':200,'15':500}
check_dic={'3':150,'6':150,'8':150,'10':300,'15':300}
#check_dic={'3':150,'6':150,'8':150,'10':150,'15':150}

num_inst=str(8)
inst_nr=str(4)
for num_inst in sizes:
    results={}
    for inst_nr in range(20,40):
        inst_nr=str(inst_nr)
        num_inst=str(num_inst)
        train_instance='/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+inst_nr+'_inst.json'
        env=jssp_light_obs_wrapper_no_action_mask(jssp_light_obs_wrapper_multi_instances(instances_list=[train_instance]))
        
        for checkpoint_nr in range(1,check_dic[num_inst]+1):
            #print('/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/oneenvironment_punish_v2ima_8_8_ppo/checkpoint-'+str(_))
            trainer.load_checkpoint('/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/oneenvironment_punish_v4_t_tima_'+num_inst+'_'+num_inst+'_ppo/checkpoint-'+str(checkpoint_nr))
            #trainer.save_checkpoint("ppo")
            episode_reward = 0
            #env=jssp_light_obs_wrapper_multi_instances(instances_list=[inst])
            #env=jssp_light_obs_wrapper_no_action_mask(jssp_light_obs_wrapper_multi_instances(instances_list=[inst]))
            #env=env_creator("s")
            #env=jssp_light_obs_wrapper_no_action_mask(jssp_klagenfurt_obs_wrapper(gym.make('jss-v1', env_config={'instance_path': 'resources/jsp_instances/standard/la01.txt'})))
            done = False
            obs = env.reset()
            iterations=0

            action_list=[]
            while not done:
                policy = trainer.get_policy()
                action, _lk, info = policy.compute_single_action(obs)
                logits = info['action_dist_inputs']
                action_mask = env.get_legal_actions().astype(np.float32)
                action_mask[action_mask==0] = - np.inf
                probs=np.multiply(logits,action_mask)
                probs[probs==np.inf]=-np.inf
                best_action_id = probs.argmax()  # deterministic
                #best_action_id = logits.argmax()
                #best_action_id = trainer.compute_single_action(obs)
                obs, reward, done, info = env.step(best_action_id)
                action_list.append(best_action_id)
                episode_reward += reward
                iterations += 1

            results['inst_'+inst_nr+'checkpoint_'+str(checkpoint_nr)]={'instance':inst_nr, 'train_reward':reward,'every_action':action_list}
    print('writing:')
    df=pd.DataFrame.from_dict(results)
    df.to_csv(f"eval_ppo_punish_size{num_inst}_inst_20_40_v3.csv")