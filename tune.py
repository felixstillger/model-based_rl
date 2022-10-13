from ray import rllib, tune
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.tune.registry import register_env
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
import gym
#from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer

from src.jss_lite.jss_lite import jss_lite
ModelCatalog.register_custom_model("dense_model", DenseModel)
from copy import deepcopy
import numpy as np


# here goes the easy observation wrapper

# from wrapper.jssplight_wrapper import jssp_light_obs_wrapper
# instance_path='/resources/jsp_instances/standard/ft06.txt'

# checkpoint_path='/training_checkpoints/checkpoints_tune'

# def env_creator(config):
#     env = jssp_light_obs_wrapper(jss_lite(instance_path=instance_path))
#     return env
import os
#print((os.path.dirname(__file__)))
curr_dir=(os.path.dirname(__file__))

instance_list=['/resources/jsp_instances/standard/la01.txt','/resources/jsp_instances/standard/la02.txt','/resources/jsp_instances/standard/la03.txt','/resources/jsp_instances/standard/la04.txt','/resources/jsp_instances/standard/la05.txt']
instance_list=[curr_dir + s for s in instance_list]

checkpoint_path='/training_checkpoints/checkpoints_tune'

from wrapper.jssplight_wrapper import jssp_light_obs_wrapper_multi_instances

def env_creator(config):
    #env = jssp_light_obs_wrapper(jss_lite(instance_path=instance_path))
    env=jssp_light_obs_wrapper_multi_instances(instances_list=instance_list)
    return env

ModelCatalog.register_custom_model("dense_model", DenseModel)    

# use tune to register the custom environment for the ppo trainer
tune.register_env('custom_jssp',env_creator)

tune.run(
    "contrib/AlphaZero",
    stop={"training_iteration": 500},
    local_dir="training_checkpoints/checkpoints_tune",
    max_failures=0,
    checkpoint_freq = 1,
    config={
        "env": 'custom_jssp',
        "disable_env_checking":True,
        "num_workers": 6,
        "rollout_fragment_length": 50,
        "train_batch_size": 50,
        "sgd_minibatch_size": 32,
        "lr": 1e-4,
        "horizon": 1000,
        "num_sgd_iter": 1,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 100,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": True,
            "add_dirichlet_noise": False,
        },
        "ranked_rewards": {
            "enable": True,
        },
        "model": {
            "custom_model": "dense_model",
        },
        "evaluation_interval": 0,
        "evaluation_config": {
            "render_env": True,
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
            },
        },
    },
)

def train(self, stop_criteria):
    """
    Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
    :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    analysis = ray.tune.run("contrib/AlphaZero", config=self.config, local_dir=self.save_dir, stop=stop_criteria,
                            checkpoint_at_end=True)
    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                       metric='episode_reward_mean')
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    return checkpoint_path, analysis

def load(self, path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
    self.agent.restore(path)

def test(self):
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    env = self.env_class(self.env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward


"""
def train_az(config, reporter):
    agent = AlphaZeroTrainer( config=config, env='custom_jssp')
    #agent.restore("/path/checkpoint_41/checkpoint-41") #continue training
    #training curriculum, start with phase 0
    phase = 0
    agent.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(phase)))
    episodes = 0
    i = 0
    while True:
        result = agent.train()
        if reporter is None:
            continue
        else:
            reporter(**result)
        if i % 10 == 0: #save every 10th training iteration
            agent.save_checkpoint(checkpoint_path)
        i+=1


trainingSteps = 100
trials = tune.run(train_az,
        config={
        "env": 'custom_jssp',
        "num_workers": 3,
        "rollout_fragment_length": 50,
        "train_batch_size": 50,
        "sgd_minibatch_size": 32,
        "lr": 1e-4,
        "num_sgd_iter": 1,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 100,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        },
        "ranked_rewards": {
            "enable": True,
        },
        "model": {
            "custom_model": "dense_model",
        },
        "evaluation_interval": 1,
        "evaluation_config": {
            "render_env": True,
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
            },
        }
        },
        local_dir="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/checkpoints_tune",
        resources_per_trial={
            "cpu": 7,
            "gpu": 0,
            "extra_cpu": 0,
        },
        stop={
            "training_iteration": trainingSteps,
        },
         )
#return_trials=True
""" 
"""
tune.run(
    agent,
    stop={"training_iteration": 500},
    local_dir="/Users/felix/sciebo/masterarbeit/progra/model-based_rl/training_checkpoints/checkpoints_tune",
    max_failures=0,
    config={
        "env": 'custom_jssp',
        "num_workers": 6,
        "rollout_fragment_length": 50,
        "train_batch_size": 50,
        "sgd_minibatch_size": 32,
        "lr": 1e-4,
        "horizon": 1000,
        "num_sgd_iter": 1,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 100,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.20,
            "dirichlet_noise": 0.03,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
        },
        "ranked_rewards": {
            "enable": True,
        },
        "model": {
            "custom_model": "dense_model",
        },
        "evaluation_interval": 0,
        "evaluation_config": {
            "render_env": True,
            "mcts_config": {
                "argmax_tree_policy": True,
                "add_dirichlet_noise": False,
            },
        },
    },
)
"""