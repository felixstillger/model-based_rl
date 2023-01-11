# %%
from ray import rllib, tune
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
#from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from src.jss_lite.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
import gym
from src.jss_lite.jss_lite import jss_lite
ModelCatalog.register_custom_model("dense_model", DenseModel)
from copy import deepcopy
import numpy as np

train_agent=True
config = {
    "framework": "torch",
    "disable_env_checking":False,
    "num_workers"       : 6,
    "rollout_fragment_length": 50,
    "train_batch_size"  : 500,
    "sgd_minibatch_size": 64,
    "lr"                : 0.0001,
    "horizon"           : 600,
    "soft_horizon"      : True,
    "num_sgd_iter"      : 1,
    "horizon"           : 100,
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

from wrapper.jssplight_wrapper import jssp_light_obs_wrapper


def env_creator(config):
    env = jssp_light_obs_wrapper(jss_lite(instance_path='resources/jsp_instances/standard/ft06.txt'))
    return env

ModelCatalog.register_custom_model("dense_model", DenseModel)    

# use tune to register the custom environment for the ppo trainer
tune.register_env('custom_jssp',env_creator)



# %%
agent = AlphaZeroTrainer( config=config, env='custom_jssp')
import time
if train_agent:
    # checkpoint_path = analysis.get_last_checkpoint() or args.checkpoint
    ## use string number to restore pre trained agent
    # nr_restore="10"
    #checkpoint_path=f'checkpoints_az/rllib_checkpoint{nr_restore}/checkpoint_{nr_restore.zfill(6)}/checkpoint-{nr_restore}'
    #agent.load_checkpoint("checkpoints_az/rllib_checkpoint1")
    #print("awd")
    #agent.restore("checkpoints_az/rllib_checkpoint1/checkpoint_000001/checkpoint-1")
    #agent.load_checkpoint("published_checkpoints/az_taxi/checkpoint-34")
    print("start training")
    for _ in range(0,100):
        t=time.time()
        agent.train()
        print(f"training iteration {_} finished after {time.time()-t} seconds")
        agent.save_checkpoint(f"training_checkpoints/checkpoints_az_jsslite")
    


# %%
import time
length_list=[]
reward_list=[]
for _ in range(10):
    policy = agent.get_policy(DEFAULT_POLICY_ID)
    action_list=[]
    env = jss_lite(instance_path='resources/jsp_instances/standard/ft06.txt')

    obs = env.reset()
    # env2 is copy for later going evaluation
    env2=deepcopy(env)

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
        action, _, _ = policy.compute_single_action(obs, episode=episode)
        action_list.append(action)
        #print(action_dic[action])
        obs, reward, done, _ = env.step(action)
        #print(obs)
        #env.render(render_mode='human')
        #time.sleep(0.1)
        episode.length += 1

    length_list.append(episode.length)
    reward_list.append(reward)
    env.close()

print(done)
print(f"reward is {reward}")
print(f"invalid action count:{env.invalid_actions}")



