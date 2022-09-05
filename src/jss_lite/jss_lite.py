from turtle import shape
import pandas as pd
import gym
import numpy as np
from pathlib import Path
from os.path import exists
from math import ceil

class jss_lite(gym.Env):

    def __init__(self, instance_path=None):
        #allocate parameters:
        self.n_jobs=None
        self.n_machines=None
        self.n_tasks=None
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
        self.observation_space_shape=(self.n_jobs+1,6)
        #' gym relevant stuff:
        # action space contains assignment of task from job to machine
        self.action_space = gym.spaces.Discrete(self.n_jobs+1) 
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
        ######## inital observation
        self.observation= np.zeros(shape=self.observation_space_shape)
        # first row is legal action mask
        self.count_finished_tasks_job_matrix=np.zeros(self.n_jobs,dtype=np.int32)
        self.current_timestep=0

        ## todo:assure that tasks is right!
        self.n_tasks=self.job_tasklength_matrix.shape[1]

        self.longest_tasklength=np.amax(self.job_tasklength_matrix)
        # first entry stores active processed time of job, second entry overall task time
        self.processed_and_max_time_job_matrix=np.zeros((self.n_jobs,2))
        # machine status stores: current job, time operation/task left, total time working
        self.current_machines_status=np.zeros((self.n_machines,3))

        for i in range(self.n_jobs):
            self.processed_and_max_time_job_matrix[i][1]=sum(self.job_tasklength_matrix[i])
        # observation space contains for every job:
                    #   - task can be assigned (previouse task of job is finished! and machine is free -> legal action mask)
                    #   - process time for next task // normalized by longest task over all jobs
                    #   - seconds this job needs in best case // normalized by longest job

                    #   - earliest time next task from different job needs this machine
                    #   - smth. like % of next task or duration of task

                # observation space contains for every machine:
                    #   - time which this machine has to be running in total // normalized by all jobs: beginning-> 1, finish ->0
                    #   -             
        """ ideas from klagenfurt environment
            -Legal job
                    -Left over time on the current op
                    -Current operation %
                    -Total left over time
                    -When next machine available
                    -Time since IDLE: 0 if not available, time otherwise
                    -Total IDLE time in the schedule
        """

        # inital observation
        self.observation= np.zeros(shape=self.observation_space_shape)
        # first row is legal action mask
        self.observation[:,0]=np.full((self.n_jobs+1,),True)
        # set skip timestep to False
        self.observation[self.n_jobs][0]=False

        for i in range(self.n_jobs):
            # second attribute: current time left on current task:
            self.observation[i,1]=self.norm_with_max(0,self.longest_tasklength)
            # third attribute: process time of next scheduled task
            self.observation[i,2]=self.norm_with_max(self.job_tasklength_matrix[i][self.count_finished_tasks_job_matrix[i]],self.longest_tasklength)
            #fourth attribute: process on current job in percent
            self.observation[i,3]=self.norm_with_max(self.processed_and_max_time_job_matrix[i][0],self.processed_and_max_time_job_matrix[i][1])
            # time to next machine available
            self.observation[i][4]=self.norm_with_max(self.current_machines_status[i][1],self.longest_tasklength)
            # count finished tasks// normalized
            self.observation[i][5]=self.norm_with_max(self.count_finished_tasks_job_matrix[i],self.n_tasks)



                        
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

    def get_legal_actions(self,obs):
        action_mask=np.full((self.n_jobs+1,),False)
        avail_machines=[]
        for i in range(self.n_machines):
            if self.current_machines_status[i][0]==np.nan:
                avail_machines.append(i)
        for i in range(self.n_jobs):
            #obs i 5 is normed count of finished tasks
            if self.job_machine_matrix[i][self.denorm_with_max(obs[i][5],self.n_tasks)] in avail_machines:
                action_mask[i]=True

    def norm_with_max(value,max_value):
        # just a function to norm towards a max value to use the same round metric and to get values between 0 and 1
        return round(value/max_value,4)

    def denorm_with_max(normed_value,max_value):
        return ceil(normed_value*max_value)
