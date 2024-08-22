#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment_CU import Env_cellular
from DQN_agent_pytorch import DQNAgent
import time
import torch
fd = 10
Ts = 20e-3
n_x = 5
n_y = 5
L = 2
C = 16
maxM = 4   # user number in one BS
min_dis = 0.01 #km
max_dis = 1. #km
max_p = 38. #dBm
min_p = 5
p_n = -114. #dBm
power_num = 10 #action_num
seed=11
Ns=21 # 11
max_episode=100 #for test

env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env.set_Ns(Ns)

batch_size = 500
buffer_size = 50000
INITIAL_EPSILON = 0.2 
FINAL_EPSILON = 0.0001
state_num = env.state_num
action_num = env.power_num  
num_of_agents=n_x*n_y
a=np.zeros(n_x*n_y*maxM).astype(np.int64)
p=np.zeros(n_x*n_y*maxM)

dqn_agent=DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0, FINAL_EPSILON=0,max_episode=5000,
                 replace=1000)

l=max_episode*(Ns-1)



result_path = 'test/result/DQN/DQN.txt'

choice=3
# Central 1
# Dist 2
# FL 3
# Personal 4

############################################################################################################################


if choice==1: # Central
    executaion_time_cost=0
    t=time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dqn_agent.load_models('models_param/DQN/Central_DQN.pth')
    agent=dqn_agent
    reward_hist_dqn = list()   
    for k in range(max_episode):
        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            st=time.time()
            a = agent.select_action(s_actor,k)
            executaion_time_cost=executaion_time_cost+time.time()-st
            p=env.get_power_set(min_p)[a]
            s_actor_next, _, rate, r , _= env.step(p)
            s_actor = s_actor_next
            reward_hist_dqn.append(r)
    reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
    DQN_cent_mean=np.nanmean(reward_hist_dqn)
    DQN_cent_std=np.nanstd(reward_hist_dqn)
    print("Central_DQN : Mean = %.4f , STD = %.4f "%(DQN_cent_mean,DQN_cent_std))
    print("elapsed time : %2f"%(time.time()-t))
    print("executaion_time_cost :",executaion_time_cost/(l))
    with open(result_path, 'a') as f:
        f.write("Central_DQN : Mean = %.4f , STD = %.4f \n"%(DQN_cent_mean,DQN_cent_std))
        f.write("Elapsed time : %2f \n"%(time.time()-t))
        f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l)))


elif choice==2: # Dist
    Agents_list=[]
    for n in range(num_of_agents):
        Agents_list.append(DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.0, FINAL_EPSILON=0.0,max_episode=5000,
                replace=1000)) # gamma=0

    for i in range(num_of_agents):
        Agents_list[i].load_models('test/models/DQN/Dist/Distributed_DQN_index_'+str(n)+'.pth')

    
    executaion_time_cost=0
    t=time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    reward_hist_dqn = list()                
    for k in range(max_episode):
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                agent=Agents_list[n]

                st=time.time()
                a_agent = agent.select_action(s_actor_agent,k) # 
                executaion_time_cost=executaion_time_cost+time.time()-st       
                p_agent=env.get_power_set(min_p)[a_agent]       
                a[n*maxM:(n+1)*maxM]=a_agent
                p[n*maxM:(n+1)*maxM]=p_agent
            s_actor_next, _, rate, r,rate_all_agents = env.step(p)
            s_actor = s_actor_next
            reward_hist_dqn.append(r)  
    reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
    DQN_dist_mean=np.nanmean(reward_hist_dqn)
    DQN_dist_std=np.nanstd(reward_hist_dqn)
    print("Distributed_DQN : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
    print("elapsed time : %2f"%(time.time()-t))
    print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
    with open(result_path, 'a') as f:
        f.write("Distributed__DQN : Mean = %.4f , STD = %.4f \n"%(DQN_dist_mean,DQN_dist_std))
        f.write("Elapsed time : %2f \n"%(time.time()-t))
        f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l*num_of_agents)))


elif choice==3: # FL
    method="Global_Aggr_2Cluster_"
    file_name=method+"FL_DQN_Fed_AggPer"
    #AggPerNum=['1','10','100','1000']
    AggPerNum=['10']


    N=1
    

    for agg in range(1):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #dqn_agent.load_models('CFDQN/models_param/CFDQN/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        dqn_agent.load_models('models_param/CFDQN/'+file_name+'_'+AggPerNum[agg]+'_(1).pth')
        agent=dqn_agent
        reward_hist_dqn = list()                
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
                    st=time.time()
                    a_agent = agent.select_action(s_actor_agent,k)
                    executaion_time_cost=executaion_time_cost+time.time()-st       
                    p_agent=env.get_power_set(min_p)[a_agent]       
                    a[n*maxM:(n+1)*maxM]=a_agent
                    p[n*maxM:(n+1)*maxM]=p_agent
                s_actor_next, _, rate, r,rate_all_agents = env.step(p)
                s_actor = s_actor_next
                reward_hist_dqn.append(r)  
        reward_hist_dqn = np.reshape(reward_hist_dqn, [max_episode*( Ns-1)])
        DQN_dist_mean=np.nanmean(reward_hist_dqn)
        DQN_dist_std=np.nanstd(reward_hist_dqn)
        print(file_name+'_'+AggPerNum[agg]+"_(1) : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write(file_name+'_'+AggPerNum[agg]+"_("+str(N)+") : Mean = %.4f , STD = %.4f \n"%(DQN_dist_mean,DQN_dist_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")

