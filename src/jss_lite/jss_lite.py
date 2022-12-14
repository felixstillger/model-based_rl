from multiprocessing.sharedctypes import Value
import pandas as pd
import gym
import numpy as np
from os.path import exists
import os
import math
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from collections import OrderedDict
import time
import json

from copy import deepcopy
class jss_lite(gym.Env):

    def __init__(self, instance_path=None,reward_mode='optimality gap'):
        #allocate parameters:
        self.n_jobs=None
        self.n_machines=None
        self.n_tasks=None
        self.job_machine_matrix=None
        self.job_tasklength_matrix=None
        self.job_length_vector=None
        self.observation_space=None
        self.instance=None
        self.optimal_value=None
        self.horizon=None
        
        # stores all timesteps
        self.timesteps_list=[]
        self.done=False
        # parameter to save current observation
        self.observation=None
        # list for actions which are blocked for the current timestep
        self.blocked_actions=[]
        #check if correct instance path is provided and if standard convention of the instance file is followed:
        # counter for invalid actions:
        self.invalid_actions=0
        # possible modes: "makespan" or "utalisation"; prezized in reward method that returns reward
        self.reward_mode=reward_mode

        # here comes the instance loader:
        if not exists(instance_path):
            raise FileExistsError(f"File does not exists or {instance_path} is no valid path")
        if instance_path is None:
            raise ValueError("no instance is given")
        else:
            pass
        self.instance=instance_path.replace('/', ' ').split(' ')[-1].split('.')[-2]
        if any(x in self.instance for x in ["abz","dmu","yn","ta","swv","orb","la","ft"]):

            #self.instance=instance_path.replace('/', ' ').split(' ')[-1].split('.')[-2]
            # 
            curr_dir=(os.path.join(os.path.dirname(__file__),'..','..'))
            df=pd.read_csv(curr_dir+'/resources/jps_instances_metadata/instances_metadata.csv',index_col='Unnamed: 0')
            #df=pd.read_csv('/Users/felix/sciebo/masterarbeit/progra/model-based_rl/resources/jps_instances_metadata/instances_metadata.csv',index_col='Unnamed: 0')
            if self.instance in df.index:
                if not math.isnan((df['Optimal value'][self.instance])):
                    self.optimal_value=(df['Optimal value'][self.instance])
                elif not math.isnan((df['Upper bound'][self.instance])):
                    self.optimal_value=(df['Upper bound'][self.instance])
                else:
                    # optimal value goes to 0 and reward function only gets negative values
                    self.optimal_value=0
            else:
                print("Key error in metadata; key does not exists")
                self.optimal_value=0
            # here begins instance parser
            n_line = 0
            with open(instance_path, 'r') as text:
                for line_txt in text.readlines():
                    n_line+=1
                    if n_line == 1:
                        self.n_jobs,self.n_machines=map(int,line_txt.split())
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
        elif any(x in self.instance for x in ['_inst']):
            #here goes the custom ima instance parser
            with open(instance_path) as f:
                instance_dict = json.load(f)
            #self.instance=instance_path
            self.optimal_value=instance_dict['optimal_time']
            self.n_jobs=instance_dict['n_jobs']
            self.n_machines=instance_dict['n_resources']
            # colides with later assignment:
            self.n_tasks=instance_dict['n_ops_per_job']

            # matrix contains machine and coresponding job lenth
            self.job_machine_matrix = np.zeros((self.n_jobs,self.n_machines), dtype=(np.int32))
            self.job_tasklength_matrix=np.zeros((self.n_jobs,self.n_machines), dtype=(np.int32))
            # contains time to complete jobs; unused 
            self.job_length_vector = np.zeros(self.n_jobs, dtype=np.int32)

            for job in range(self.n_jobs):
                self.job_machine_matrix[job,:]= instance_dict['jssp_instance']['machines'][job]
            for machine in range(self.n_machines):
                self.job_tasklength_matrix[machine,:]= instance_dict['jssp_instance']['durations'][machine]

        else:
            raise NotImplementedError("till now only standard instances are implemented")

        #todo define observation space:
        # states are order of jobs to machines
        self.observation_space_shape=(max(self.n_jobs,self.n_machines),7)
        # for every job an assignment to a machine, and for every job a dummy assignment to block the machine by now
        self.action_space_shape=(2*self.n_jobs)
        #' gym relevant stuff:
        # action space contains assignment of task from job to machine
        self.action_space = gym.spaces.Discrete(self.action_space_shape)         
        self.observation_space = gym.spaces.Dict({
                "action_mask": gym.spaces.Box(0,1,shape=(self.action_space.n,),dtype=np.int32),
                "obs": gym.spaces.Box(low=0.0,high=1.0,
                    shape=(max(self.n_jobs,self.n_machines)*7,),dtype=np.float64)
                     
                                                }
                                                )
        #extended parameters
        ## todo:assure that tasks is right!
        self.n_tasks=self.job_tasklength_matrix.shape[1]
        #stores quatuple(job,task,start_time,finish_time) 
        self.production_list=np.empty((self.n_machines,int(self.n_jobs*self.n_tasks/self.n_machines)),dtype=object)
        # counter for finished taska and jobs
        self.count_finished_tasks_machine_matrix=np.zeros(self.n_machines,dtype=np.int32)
        self.count_finished_tasks_job_matrix=np.zeros(self.n_jobs,dtype=np.int32)
        # starting point
        #self.timesteps_list.append(0)
    
        # initial observation
        self.observation= np.zeros(shape=self.observation_space_shape,dtype=np.float64)
        # first row is legal action mask
        #todo: check if n_jobs << n_machines
        self.observation[:self.n_jobs,0]=np.full((self.n_jobs,),True)
        # set skip waiting to False
        self.observation[:self.n_jobs,1]=np.full((self.n_jobs,),False)
        self.current_timestep=0
        self.longest_tasklength=np.amax(self.job_tasklength_matrix)
        # first entry stores active processed time of job, second entry overall task time
        self.processed_and_max_time_job_matrix=np.zeros((self.n_jobs,2))
        # machine status stores: current job, time operation/task left, total time working
        self.current_machines_status=np.zeros((self.n_machines,3))
        # set current job to nan because no job is assigned to the machines
        self.current_machines_status[:,0]=np.nan
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

        self.update_observation()
        # set actions to True
        self.observation[:self.n_jobs,0]=np.full((self.n_jobs,),True)
        # set skip waiting to False
        self.observation[:self.n_jobs,1]=np.full((self.n_jobs,),False)
    def get_reward(self,section):
        # you can implement your own reward styles; there are different sections where reward can occur; if you want to define more sections feel free in the step method
        # by now sections are: "invalid_action", "next_timestep", "on_step", "on_done"
        if self.reward_mode=='makespan':
            if section=='invalid_action':
                return 0
            elif section=='next_timestep':
                return 0
            elif section =='on_step':
                return 0
            elif section =='on_done':
                return 2*self.optimal_value-self.current_timestep
            else:
                raise ValueError(f"section: {section} ist not implemented yet")
        elif  self.reward_mode=='optimality gap':
            if section=='invalid_action':
                #print(f"Error invalid action on instance {self.instance}")
                #print()
                return -0.0000001
                #return 0
            elif section=='next_timestep':
                return 0
            elif section =='on_step':
                return 0
            elif section =='on_done':
                #return 100-(100*(self.current_timestep-self.optimal_value)/self.optimal_value)
                #scaled to values between [0,1]
                return 1-((self.current_timestep-self.optimal_value)/self.optimal_value)

            else:
                raise ValueError(f"section: {section} ist not implemented yet")
        elif self.reward_mode=='utilisation':
            pass
        else:
            raise ValueError(f"following reward mode: {self.reward_mode} is not defined")
    
        reward=self.get_reward(section='invalid_action')
    
    def step(self,action:int):
        reward=0
        # ensure to check right index for mask in observation
        if action >= self.n_jobs:
            mask_index=(action-self.n_jobs,1)
        else:
            mask_index=(action,0)
        if self.observation[mask_index]==0:
        #check here if invalid actions are done: if needed rise Error here
            self.invalid_actions+=1
            #reward=-1
            reward=self.get_reward(section='invalid_action')
        # as long timesteps list is not empty and action is a real and no dummy action; todo: implement if statement for dummy action
        elif action < self.n_jobs:
            #assure that it is no dummy job   
            # assign task (from action) to available machine and update the machine status
            # block the free machine with current job
            #todo: delete error
            #if math.isnan(self.current_machines_status[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]],0])==False:
            #    raise ValueError("machine is not free") 
            self.current_machines_status[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]],0]=action
            self.current_machines_status[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]],1]=self.job_tasklength_matrix[action,self.count_finished_tasks_job_matrix[action]]
            # update production_list:(job,tas,start,finish)
            #self.production_list[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]],self.count_finished_tasks_job_matrix[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]]]]=(action,self.count_finished_tasks_job_matrix[action],self.current_timestep,self.current_timestep+self.job_tasklength_matrix[action,self.count_finished_tasks_job_matrix[action]])
            self.production_list[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]],int(self.count_finished_tasks_machine_matrix[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]]])]=(action,self.count_finished_tasks_job_matrix[action],self.current_timestep,self.current_timestep+self.job_tasklength_matrix[action,self.count_finished_tasks_job_matrix[action]])
            # add end time to timesteps list, first check if value is not already in list:
            if self.current_timestep+self.job_tasklength_matrix[action,self.count_finished_tasks_job_matrix[action]] not in self.timesteps_list:
                self.timesteps_list.append(self.current_timestep+self.job_tasklength_matrix[action,self.count_finished_tasks_job_matrix[action]])
            #update action in observation and coresponding machine in observation:

            self.update_observation(action=action)
        else: #action is dummy action
            # get action to block, r_action 
            r_action=action-self.n_jobs
            # append action to the blocked actions
            self.blocked_actions.append(r_action)
            # add here timesteps to
            if not self.timesteps_list:
                # should never happen: todo: check if could arise:     
                self.timesteps_list.append(self.current_timestep+self.job_tasklength_matrix[r_action,self.count_finished_tasks_job_matrix[r_action]])
            #          
        # update observation:
        # update mask in observation:
        #old:
        #self.observation[:,0]=self.get_legal_actions()
        # split normal and dummy actions in two columns
        self.observation[:self.n_jobs,:2]=(np.transpose(np.reshape(self.get_legal_actions(),(2,self.n_jobs))))

        while True not in self.observation[:,:2]:
            #print("go to next timesteps possible")
            #implement to jump to next timesteps
            #check if done? todo: add aditional checks like are all tasks done and satisfied or just problems with timesteps?
            if not self.timesteps_list:
                #print("finished")
                # this is reward to mini makespan: todo:implement utilisation etc.
                for row in self.production_list:
                    if None in row:
                        print(self.count_finished_tasks_job_matrix)
                        raise ValueError("done = true but production did not finished; problem")
                #reward=-self.current_timestep
                # for ft06 optimal value is 55
                self.done=True
                reward=self.get_reward(section='on_done')
                break
            else:
                #print("update step")
                # temp. saving timestep
                forward_timestep=self.current_timestep
                self.timesteps_list.sort()
                self.current_timestep=self.timesteps_list.pop(0)
                #print(f"curr timestep: {self.current_timestep}")
                #print(f"new time is {self.current_timestep}")
                # how large is the timestep?
                forward_timestep=self.current_timestep-forward_timestep
                # substract forwarded time from current production, update on all active machines:
                # free list of blocked actions
                self.blocked_actions=[]
                for i in range(self.n_machines):
                    #check if machine is active, active if 0th entry is not nan
                    if math.isnan(self.current_machines_status[i,0])==False:
                        job=int(self.current_machines_status[i,0])
                        self.processed_and_max_time_job_matrix[job,0]=self.processed_and_max_time_job_matrix[job,0]+forward_timestep
                        # leftover time for task on machine i
                        self.current_machines_status[i,1]-=forward_timestep
                        #total time of machine working
                        self.current_machines_status[i,2]+=forward_timestep
                        #if self.processed_and_max_time_job_matrix[i,0] >= self.processed_and_max_time_job_matrix[i,1]:
                        if self.current_machines_status[i,1] == 0:
                            #machine is finished with task, need reward before
                            self.current_machines_status[i,0]=np.nan
                            # residual time of job, obviously =0
                            #if self.current_machines_status[i,1]!=0:
                            #    raise ValueError(f"Problems with left over time for task is {self.current_machines_status[i,1]} not 0") 
                            self.count_finished_tasks_job_matrix[job]+=1
                            self.count_finished_tasks_machine_matrix[i]+=1 
                # old: self.observation[:,0]=self.get_legal_actions()
                self.observation[:self.n_jobs,:2]=(np.transpose(np.reshape(self.get_legal_actions(),(2,self.n_jobs))))

        info = {
            'action': action
        }
        #self.observation[:,0]=self.get_legal_actions(self.observation)
        state=self.observation_to_state()
        ## insert: update timesteps if there is no legal action left

        return state, reward, self.done, info

    def get_state(self):
        #print("state getted")
        return deepcopy(self)

    def set_state(self, state):
        #print("state setted")
        #self= deepcopy(state)
        self=state
        self.done=state.done
        out=OrderedDict({"action_mask":np.array(self.get_legal_actions()).astype(np.int32),"obs":(np.ravel(self.observation))})

        #return OrderedDict({"action_mask":self.get_legal_actions(self.observation),"obs":(np.ravel(self.observation))})
        return out
    def render(self,x_bar="Machine",y_bar="Job",start_count=0,keep_grey=False):
        # is not time critical function. so O(n^2) is no problem; todo:make more pretty
        def production_to_dict(input,i,j):
            return(dict(Job=f"job_{str(input[0]+start_count).zfill(max_len)}", Start=dt + timedelta(seconds=int(input[2]+start_count)),Finish=dt + timedelta(seconds=int(input[3])),Machine=f"machine_{str(i+start_count).zfill(max_len)}"))
        dt = datetime(2022, 1, 1, 0, 0, 0)
        #stores quatuple(job,task,start_time,finish_time) 
        liste=[]
        max_len=len(str(max(self.production_list.shape[0]-1+start_count,self.production_list.shape[1]-1+start_count)))
        for i in range(self.production_list.shape[0]):
            for j in range(self.production_list.shape[1]):
                if self.production_list[i,j] is not None:
                    liste.append(production_to_dict(self.production_list[i,j],i,j))
        df_render=pd.DataFrame(liste)
        if df_render.empty:
            # do not rise an error; production plan is empty and could be filled
            print("Production plan is empty; nothing to plot")
        else:
            df_render.sort_values(by=y_bar,inplace=True)
            #fig = px.timeline(df_render, x_start="Start", x_end="Finish", y="Task", color="Resource")
            #colour in grey:
            if keep_grey:
                rgb_add=math.floor( 255/int(self.n_machines))
                color_map={}
                for i in range(self.n_machines):
                    color_map['machine_'+str(i)]=f"rgb({i*rgb_add},{i*rgb_add},{i*rgb_add})"
                fig = px.timeline(df_render, x_start="Start", x_end="Finish", color=x_bar, y=y_bar,color_discrete_map=color_map)
            else:
                fig = px.timeline(df_render, x_start="Start", x_end="Finish", color=x_bar, y=y_bar)

            #todo, save or do smth else to console rendering
            fig.show()

    def render_to_df(self,x_bar="Machine",y_bar="Job",start_count=0):
        # is not time critical function. so O(n^2) is no problem; todo:make more pretty
        def production_to_dict(input,i,j):
            return(dict(Job=f"job_{str(input[0]+start_count).zfill(max_len)}", Start=dt + timedelta(seconds=int(input[2]+start_count)),Finish=dt + timedelta(seconds=int(input[3])),Machine=f"machine_{str(i+start_count).zfill(max_len)}"))
        dt = datetime(2022, 1, 1, 0, 0, 0)
        #stores quatuple(job,task,start_time,finish_time) 
        liste=[]
        max_len=len(str(max(self.production_list.shape[0]-1+start_count,self.production_list.shape[1]-1+start_count)))
        for i in range(self.production_list.shape[0]):
            for j in range(self.production_list.shape[1]):
                if self.production_list[i,j] is not None:
                    liste.append(production_to_dict(self.production_list[i,j],i,j))
        df_render=pd.DataFrame(liste)
        df_render.sort_values(by=y_bar,inplace=True)
        #fig = px.timeline(df_render, x_start="Start", x_end="Finish", y="Task", color="Resource")
        fig = px.timeline(df_render, x_start="Start", x_end="Finish", color=x_bar, y=y_bar)
        return df_render, fig

    def get_legal_actions(self):
        action_mask=np.full((2*self.n_jobs,),0,dtype=np.int32)
        # get current available machines
        avail_machines=[i for i in range(self.n_machines) if math.isnan(self.current_machines_status[i][0]) ]
        for i in range(self.n_jobs):
            if self.count_finished_tasks_job_matrix[i]!=self.n_tasks:
                if self.job_machine_matrix[i][self.count_finished_tasks_job_matrix[i]] in avail_machines and i not in self.current_machines_status[:,0] and i not in self.blocked_actions: 
                    #True
                    action_mask[i]=1
                    # here goes the dummy tasks
                    finished_tasks=[]
                    action_process_time=self.job_tasklength_matrix[i,self.count_finished_tasks_job_matrix[i]]
                    for j in range(self.n_machines):
                        if self.current_machines_status[j,1]>0 and self.current_machines_status[j,1] < action_process_time:
                            finished_tasks.append(self.current_machines_status[j,0])
                    for j in finished_tasks:
                        j=int(j)
                        if self.count_finished_tasks_job_matrix[j] < self.n_tasks-1:
                            ## todo comment and rework
                            if self.job_machine_matrix[i,self.count_finished_tasks_job_matrix[i]]== self.job_machine_matrix[j,self.count_finished_tasks_job_matrix[j]+1]:
                                #True
                                if j != i:
                                    action_mask[i+self.n_jobs]=1                             
        return action_mask

    
    def get_legal_actions_old(self):
        action_mask=np.full((2*self.n_jobs,),0,dtype=np.int32)
        avail_machines=[]
        for i in range(self.n_machines):
            if math.isnan(self.current_machines_status[i][0]):
                avail_machines.append(i)
        for i in range(self.n_jobs):
            if self.count_finished_tasks_job_matrix[i]!=self.n_tasks:
                if self.job_machine_matrix[i][self.count_finished_tasks_job_matrix[i]] in avail_machines and i not in self.current_machines_status[:,0] and i not in self.blocked_actions: 
                    #True
                    action_mask[i]=1
                    # here goes the dummy tasks
                    finished_tasks=[]
                    action_process_time=self.job_tasklength_matrix[i,self.count_finished_tasks_job_matrix[i]]
                    action_mask[i+self.n_jobs]=1
                    for j in range(self.n_machines):
                        if self.current_machines_status[j,1]>0 and self.current_machines_status[j,1] < action_process_time:
                            finished_tasks.append(self.current_machines_status[j,0])
                    for j in finished_tasks:
                        j=int(j)
                        if self.count_finished_tasks_job_matrix[j] < self.n_tasks-1:
                            ## todo comment and rework
                            if self.job_machine_matrix[i,self.count_finished_tasks_job_matrix[i]]== self.job_machine_matrix[j,self.count_finished_tasks_job_matrix[j]+1]:
                                #True
                                if j != i:
                                    action_mask[i+self.n_jobs]=1                             
        return action_mask


    def norm_with_max(self,value,max_value):
        # just a function to norm towards a max value to use the same round metric and to get values between 0 and 1
        if max_value==0:
            return 0
        else:
            out=round(value/max_value,4)
            if out > 1:
                print(value)
                print(max_value)
                raise ValueError("normalize did not work")
            return out

    def denorm_with_max(self,normed_value,max_value):
        return math.ceil(normed_value*max_value)
        
    def observation_to_state(self):
        # here comes the transformation to returned observation 
        #todo: complete function
        status= {"obs":((np.ravel(self.observation)).astype(np.float32)),"action_mask":np.array(self.get_legal_actions()).astype(np.int32)}
        return status
    # debugging method to check the plausiblity of the production plan
    def check_production_plan(self):
        pass
    def update_observation(self,action=None):
        # observation space contains for every job:
            #   - task can be assigned (previouse task of job is finished! and machine is free -> legal action mask)
            #   - process time for next task // normalized by longest task over all jobs
            #   - seconds this job needs in best case // normalized by longest job
            #   - earliest time next task from different job needs this machine
            #   - smth. like % of next task or duration of task
        # observation space contains for every machine:
            #   - time which this machine has to be running in total // normalized by all jobs: beginning-> 1, finish ->0
            #   -   max(2*self.n_jobs,self.n_machines)*6  
        #old: self.observation[:,0]=self.get_legal_actions()
        self.observation[:self.n_jobs,:2]=(np.transpose(np.reshape(self.get_legal_actions(),(2,self.n_jobs))))

        # decide here if you want to update whole observation or just the one influenced by action
        if action ==None:
            first_iter=range(self.n_jobs)
            second_iter=range(self.n_machines)
        else:
            first_iter=[action]
            second_iter=[self.job_machine_matrix[action,self.count_finished_tasks_job_matrix[action]]]
        for i in first_iter:
            # second attribute: current time left on current task:
            self.observation[i,2]=self.norm_with_max(0,self.longest_tasklength)
            # third attribute: process time of next scheduled task
            if self.count_finished_tasks_job_matrix[i]!=self.n_tasks:
                self.observation[i,3]=self.norm_with_max(self.job_tasklength_matrix[i][self.count_finished_tasks_job_matrix[i]],self.longest_tasklength)
            else:
                self.observation[i,3]= 0
            #fourth attribute: process on current job in percent
            self.observation[i,4]=self.norm_with_max(self.processed_and_max_time_job_matrix[i][0],self.processed_and_max_time_job_matrix[i][1])

            # count finished tasks// normalized
            self.observation[i][5]=self.norm_with_max(self.count_finished_tasks_job_matrix[i],self.n_tasks)
        for i in second_iter:
            # time to next machine available
            self.observation[i][6]=self.norm_with_max(self.current_machines_status[i][1],self.longest_tasklength)
            # second attribute: current time left on current task, if task is not assigned it got value 0: todo: checkout it 0 or full time makes sense
            if math.isnan(self.current_machines_status[i,0])==False:
                #print(self.current_machines_status[i,0])
                self.observation[int(self.current_machines_status[i,0]),2]=self.norm_with_max(self.current_machines_status[i][1],self.longest_tasklength)
          

    def reset(self):
        #print("state resetet")
        self.blocked_actions=[]
        self.timesteps_list=[]
        self.done=False
        #stores quatuple(job,task,start_time,finish_time) 
        self.production_list=np.empty((self.n_machines,int(self.n_tasks*self.n_jobs/self.n_machines)),dtype=object)
        # counter for finished taska and jobs
        self.count_finished_tasks_machine_matrix=np.zeros(self.n_machines)
        self.count_finished_tasks_job_matrix=np.zeros(self.n_jobs,dtype=np.int32)
        # starting point
        # parameter to save current observation
        self.observation=None
        # reset current machine status:
        self.current_machines_status=np.zeros((self.n_machines,3))
        # set current job to nan because no job is assigned to the machines
        self.current_machines_status[:,0]=np.nan
        self.observation= np.zeros(shape=self.observation_space_shape,dtype=np.float64)
        self.invalid_actions=0
        # reset observation
        # first row is legal action mask
        #todo: check if n_jobs << n_machines

        self.current_timestep=0
        ## todo:assure that tasks is right!
        self.n_tasks=self.job_tasklength_matrix.shape[1]
        self.longest_tasklength=np.amax(self.job_tasklength_matrix)
        # first entry stores active processed time of job, second entry overall task time
        self.processed_and_max_time_job_matrix=np.zeros((self.n_jobs,2))
        self.update_observation()
        #old:
        #self.observation[0:2*self.n_jobs,0]=np.full((2*self.n_jobs,),True)
        # set dummy actions to False
        #self.observation[self.n_jobs:][0]=False
        self.observation[:self.n_jobs,0]=np.full((self.n_jobs,),True)
        # set skip waiting to False
        self.observation[:self.n_jobs,1]=np.full((self.n_jobs,),False)

        return self.observation_to_state()
        #return {"obs":(np.ravel(self.observation)).astype(np.float32),"action_mask":np.array(self.get_legal_actions(self.observation)).astype(np.int32)}

    def instance_loader(self):
        
        pass