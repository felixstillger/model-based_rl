from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import gym
from src.jss_lite.jss_lite import jss_lite
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances
from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_no_action_mask
from ray import tune
import os

train_agent=True

curr_dir='/Users/felix/sciebo/masterarbeit/progra/model-based_rl'
curr_dir=(os.path.dirname(__file__))
x=[10,15]
for ins in x:
    num_inst=str(ins)
    instances_names='ima_'+num_inst+'_'+num_inst+'_ppo'
    ima_inst_train=[]
    ima_inst_test=[]

    for i in range(0,20):
        ima_inst_train.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')
    for i in range(21,30):
        ima_inst_test.append(curr_dir+'/resources/jsp_instances/ima/'+num_inst+'x'+num_inst+'x'+num_inst+'/'+num_inst+'x'+num_inst+'_'+str(i)+'_inst.json')


    def env_creator_random_instance(config_random):
        #env= jssp_light_obs_wrapper_multi_instances(instances_list=[loading_instance],env_config=config)
        env= jssp_light_obs_wrapper_no_action_mask(jssp_light_obs_wrapper_multi_instances(instances_list=[ima_inst_train[4]],env_config=config_random))
        return env


    tune.register_env('custom_jssp',lambda config: env_creator_random_instance(config))


    restore_agent= False
    num_episodes = 1000
    #restore_path= 'training_checkpoints/checkpoints_az_jsslite/checkpoint-6'

    config = {
        # Environment (RLlib understands openAI gym registered strings).
        #"env": "CartPole-v1",
        "env": "custom_jssp",
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 5,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "tf",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        #"horizon":1000,
        "evaluation_duration":10,
        "model": {
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
        },
        # Set up a separate evaluation worker set for the
        # `trainer.evaluate()` call after training (see below).
        "evaluation_num_workers": 0,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": False,
        },
        "horizon":1000,
        "no_done_at_end":True, 
        "soft_horizon":True,
        "explore":True,
        "num_sgd_iter" : 30,
        "gamma":0.9
        
        
    }

    trainer = PPOTrainer(config=config)

    s_path=(curr_dir+'/training_checkpoints'+"/"+'oneenvironment_punish_v4_t_t'+instances_names)
    if not os.path.exists(s_path):
        os.mkdir(s_path)
    for _ in range(300):
        print(_)
        trainer.train()
        trainer.save_checkpoint(s_path)