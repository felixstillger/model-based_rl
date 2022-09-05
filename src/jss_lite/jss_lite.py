from turtle import shape
import pandas as pd
import gym
import numpy as np
from pathlib import Path
from os.path import exists

class jss_lite(gym.Env):

    def __init__(self, instance_path=None):
        #allocate parameters:
        self.n_jobs=None
        self.n_machines=None
        self.job_machine_matrix=None
        self.job_tasklength_matrix=None
        self.job_length_vector=None
        self.observation_space=None

        # parameter to save current observation
        self.observation=None
        # schedule plan for visualisation
        self.schedule_plan=None

        #check if correct instance path is provided and if standard convention of the instance file is followed:

        #get_self: n_jobs,n_machines
        instance_path = "/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jsp_instances/standard/abz5.txt"
        if not exists(instance_path):
            raise FileExistsError(f"File does not exists or {instance_path} is no valid path")
        if instance_path is None:
            raise ValueError("no instance is given")
            # here begins instance parser
        if any(x in instance_path for x in ["abz","dmu","yn","ta","swv","orb","la","ft"]):
            n_line = 0
            with open(instance_path, 'r') as text:
                for line_txt in text.readlines():
                    n_line+=1
                    if n_line == 1:
                        self.n_jobs,n_machines=map(int,line_txt.split())
                        # matrix contains machine and coresponding job lenth
                        self.job_machine_matrix = np.zeros((self.n_jobs,self.n_machines), dtype=(np.int32))
                        self.job_tasklength_matrix=np.zeros((self.n_jobs,self.n_machines), dtype=(np.int32))
                        # contains time to complete jobs
                        self.job_length_vector = np.zeros(self.n_jobs, dtype=np.int32)
                    else:
                        j=0
                        nr_job=n_line-1
                        cum_jobtime=0
                        for i in line_txt.split("	"):
                            i=int(i)
                            if j%2==0:
                                self.job_machine_matrix[nr_job-1][int((j)/2)]= i 
                            else:
                                self.job_tasklength_matrix[nr_job-1][int((j-1)/2)]= i 
                                cum_jobtime+=i
                            j+=1    
                        self.job_length_vector[nr_job-1]= cum_jobtime
                #todo: implement more conventions, e.g. IMA or Taillard
        else:
            raise NotImplementedError("till now only standard instances are implemented")

        #todo define observation space:
        # states are order of jobs to machines
        self.observation_space_shape=(self.n_jobs+1,5)


        #' gym relevant stuff:
        # action space contains assignment of task from job to machine
        self.action_space = gym.spaces.Discrete(self.n_jobs*self.n_machines) 
        # observation space contains for every job:
            #   - task can be assigned (previouse task of job is finished! and machine is free -> legal action mask)
            #   - process time for next task // normalized by longest task over all jobs
            #   - seconds this job needs in best case // normalized by longest job

            #   - earliest time next task from different job needs this machine
            #   - smth. like % of next task or duration of task

        # observation space contains for every machine:
            #   - time which this machine has to be running in total // normalized by all jobs: beginning-> 1, finish ->0
            #   -             
        self.observation_space = gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.int32),
                "obs": gym.spaces.Box(low=0.0,high=1.0,
                    shape=self.observation_space_shape,dtype=np.float64)    
                                                }
                                                )
        # inital observation
        self.observation= np.zeros(shape=self.observation_space_shape)
        # first row is legal action mask
        self.observation[]

                        
    def step(self,action):
        pass
    def reset(self):
        pass
    def get_state():
        pass
    def set_state():
        pass
    def render():
        pass
    def get_legal_actions(obs,action):
        pass