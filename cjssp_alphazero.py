# %%
import src.jss_graph_env.disjunctive_graph_jss_env as jss_env
import src.jsp_instance_parser 
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
import gym
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
from wrapper.jssp_wrapper import jssp_obs_wrapper
from wrapper.jssp_wrapper import Jssp_wrapper

config = {
    "framework": "torch",
    "disable_env_checking":True,
    "num_workers"       : 0,
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
        "enable": False,
    },
    "model"             : {
        "custom_model": "dense_model",

    },
}

# %%

from wrapper.jssp_wrapper import Jssp_wrapper


def env_creator(env_config):
    #env = discretetobox(gym.make("Taxi-v3"))
    #env = gym.make('LunarLander-v2')
    path='resources/jsp_instances/standard/abz8.txt'
    curr_instance=src.jsp_instance_parser.parse_jps_standard_specification(path)
    res,std_matrix=curr_instance
    env = jssp_obs_wrapper(jss_env.DisjunctiveGraphJssEnv(res,default_visualisations='gantt_console'))
    return env
ModelCatalog.register_custom_model("dense_model", DenseModel)    
tune.register_env('customjssp',env_creator)

agent = AlphaZeroTrainer( config=config, env='customjssp')

# %%
print("start training")
for _ in range(0,150):
    agent.train()
    print(f"training iteration {_} finished")
    #agent.save(f"save_az/rllib_checkpoint{_}")
    agent.save_checkpoint(f"training_checkpoints/checkpoints_az_jsp")
    #agent.save_to_object(f"objects_az/rllib_checkpoint{_}")

# %%
import numpy as np
import random
env.reset()
action_space=env.action_space
action_mask=env.valid_action_mask()
#task_mask = self.valid_action_mask
#job_mask = np.array_split(action_mask, 10)[action]
action_list=[]

action_list=np.arange(0,env.n_jobs*env.n_machines)


for i in range(1000):
    action_mask=env.valid_action_mask()
    actions=action_list[action_mask]
    next_action=random.choice(actions)
    #next_action=actions[action_mask[0]]
    #print(next_action)
    state,reward,done,info =env.step(next_action)
    if done == True:
        print(f"finished after {i} steps")
        break
    #env.render()

env.render()

# %%
env.render()

# %%
import src.modelbased.alphazero as az

# %%


# %%
env.close()


