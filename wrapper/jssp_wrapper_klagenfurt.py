from copy import deepcopy
import bisect

import gym
import numpy as np
import src.jss_graph_env.disjunctive_graph_jss_env as jss_env
import src.jsp_instance_parser 

class jssp_klagenfurt_obs_wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #self.observation_space = gym.spaces.Box(0, 1, (self.n,))
        self.action_space = self.env.action_space
        self.observation_space =gym.spaces.Dict(
            {
                'obs':self.env.observation_space['real_obs'],
                'action_mask':self.env.observation_space['action_mask'],
            }
        )
        
    def observation(self, obs):
        #print(obs)
        return {"obs": obs['real_obs'], "action_mask": obs['action_mask']}
        #return {"obs": new_obs, "action_mask": np.array([1] * self.action_space.n, dtype=np.float32)}
        #return new_obs
    def get_state(self):
        return deepcopy(self.env)

    def set_state(self, state):
        self.env = deepcopy(state)
        return {"obs":self.env.state,"action_mask":self.env.legal_actions}

    def step(self, action: int):
        ## this if statemenet differs from original implementation, if an illegal action ist proposed, it is rewarded negativ, but the current state is not changed... hopefully does not end in a bad loop

        else:

            reward = 0.0
            if action == self.jobs:
                self.env.nb_machine_legal = 0
                self.env.nb_legal_actions = 0
                for job in range(self.env.jobs):
                    if self.env.legal_actions[job]:
                        self.env.legal_actions[job] = False
                        needed_machine = self.env.needed_machine_jobs[job]
                        self.env.machine_legal[needed_machine] = False
                        self.env.illegal_actions[needed_machine][job] = True
                        self.env.action_illegal_no_op[job] = True
                while self.env.nb_machine_legal == 0:
                    reward -= self.env._increase_time_step()
                scaled_reward = self.env._reward_scaler(reward)
                self.env._prioritization_non_final()
                self.env._check_no_op()
                return self.env._get_current_state_representation(), scaled_reward, self.env._is_done(), {}
            else:
                current_time_step_job = self.env.todo_time_step_job[action]
                machine_needed = self.env.needed_machine_jobs[action]
                time_needed = self.env.instance_matrix[action][current_time_step_job][1]
                reward += time_needed
                self.env.time_until_available_machine[machine_needed] = time_needed
                self.env.time_until_finish_current_op_jobs[action] = time_needed
                self.env.state[action][1] = time_needed / self.env.max_time_op
                to_add_time_step = self.env.current_time_step + time_needed
                if to_add_time_step not in self.env.next_time_step:
                    index = bisect.bisect_left(self.env.next_time_step, to_add_time_step)
                    self.env.next_time_step.insert(index, to_add_time_step)
                    self.env.next_jobs.insert(index, action)
                self.env.solution[action][current_time_step_job] = self.env.current_time_step
                for job in range(self.env.jobs):
                    if self.env.needed_machine_jobs[job] == machine_needed and self.env.legal_actions[job]:
                        self.env.legal_actions[job] = False
                        self.env.nb_legal_actions -= 1
                self.env.nb_machine_legal -= 1
                self.env.machine_legal[machine_needed] = False
                for job in range(self.env.jobs):
                    if self.env.illegal_actions[machine_needed][job]:
                        self.env.action_illegal_no_op[job] = False
                        self.env.illegal_actions[machine_needed][job] = False
                # if we can't allocate new job in the current timestep, we pass to the next one
                while self.env.nb_machine_legal == 0 and len(self.env.next_time_step) > 0:
                    reward -= self.env._increase_time_step()
                self.env._prioritization_non_final()
                self.env._check_no_op()
                # we then need to scale the reward
                scaled_reward = self.env._reward_scaler(reward)
                return self.env._get_current_state_representation(), scaled_reward, self.env._is_done(), {}

                
                
"""
                def step(self, action: int):
        ## this if statemenet differs from original implementation, if an illegal action ist proposed, it is rewarded negativ, but the current state is not changed... hopefully does not end in a bad loop
        if not self.env.legal_actions[action]:
            reward = -1
            return self.env._get_current_state_representation(), reward, self.env._is_done(), {}
        else:

            reward = 0.0
            if action == self.jobs:
                self.env.nb_machine_legal = 0
                self.env.nb_legal_actions = 0
                for job in range(self.env.jobs):
                    if self.env.legal_actions[job]:
                        self.env.legal_actions[job] = False
                        needed_machine = self.env.needed_machine_jobs[job]
                        self.env.machine_legal[needed_machine] = False
                        self.env.illegal_actions[needed_machine][job] = True
                        self.env.action_illegal_no_op[job] = True
                while self.env.nb_machine_legal == 0:
                    reward -= self.env._increase_time_step()
                scaled_reward = self.env._reward_scaler(reward)
                self.env._prioritization_non_final()
                self.env._check_no_op()
                return self.env._get_current_state_representation(), scaled_reward, self.env._is_done(), {}
            else:
                current_time_step_job = self.env.todo_time_step_job[action]
                machine_needed = self.env.needed_machine_jobs[action]
                time_needed = self.env.instance_matrix[action][current_time_step_job][1]
                reward += time_needed
                self.env.time_until_available_machine[machine_needed] = time_needed
                self.env.time_until_finish_current_op_jobs[action] = time_needed
                self.env.state[action][1] = time_needed / self.env.max_time_op
                to_add_time_step = self.env.current_time_step + time_needed
                if to_add_time_step not in self.env.next_time_step:
                    index = bisect.bisect_left(self.env.next_time_step, to_add_time_step)
                    self.env.next_time_step.insert(index, to_add_time_step)
                    self.env.next_jobs.insert(index, action)
                self.env.solution[action][current_time_step_job] = self.env.current_time_step
                for job in range(self.env.jobs):
                    if self.env.needed_machine_jobs[job] == machine_needed and self.env.legal_actions[job]:
                        self.env.legal_actions[job] = False
                        self.env.nb_legal_actions -= 1
                self.env.nb_machine_legal -= 1
                self.env.machine_legal[machine_needed] = False
                for job in range(self.env.jobs):
                    if self.env.illegal_actions[machine_needed][job]:
                        self.env.action_illegal_no_op[job] = False
                        self.env.illegal_actions[machine_needed][job] = False
                # if we can't allocate new job in the current timestep, we pass to the next one
                while self.env.nb_machine_legal == 0 and len(self.env.next_time_step) > 0:
                    reward -= self.env._increase_time_step()
                self.env._prioritization_non_final()
                self.env._check_no_op()
                # we then need to scale the reward
                scaled_reward = self.env._reward_scaler(reward)
                return self.env._get_current_state_representation(), scaled_reward, self.env._is_done(), {}"""


    
