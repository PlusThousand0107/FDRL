#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment_CU import Env_cellular
from Reinforce_Pytorch import PolicyGradientAgent
from Personal_Reinforce_Pytorch import PolicyGradientAgent as PersonalPG
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

Ns=21 # 11
max_episode=100 #for test

env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env.set_Ns(Ns)

batch_size = 512
buffer_size = 50000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.0001
state_num = env.state_num
action_num = env.power_num  
num_of_agents=n_x*n_y
a=np.zeros(n_x*n_y*maxM).astype(np.int64)
p=np.zeros(n_x*n_y*maxM)

DPG_agent=PolicyGradientAgent(lr=1e-3,state_dims=state_num,gamma=0.9,n_actions=action_num)

l=max_episode*(Ns-1)


result_path = 'test/result/DPG/DPG.txt'

choice=1
# Central 1
# Dist 2
# FL 3
# Personal 4

############################################################################################################################




if choice==1: # Central
    executaion_time_cost=0
    t=time.time()
    DPG_agent.load_model('models_param/DPG/Central_DPG.pth')
    agent=DPG_agent
    reward_dpg_list = list()   
    for k in range(max_episode):
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            st=time.time()
            a = agent.choose_action(s_actor)
            executaion_time_cost=executaion_time_cost+time.time()-st      
            p=env.get_power_set(min_p)[a]
            s_actor_next, _, rate, r ,_= env.step(p)
            s_actor = s_actor_next
            reward_dpg_list.append(r)
    reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
    DPG_mean=np.nanmean(reward_dpg_list)
    DPG_std=np.nanstd(reward_dpg_list)
    print("Central_DPG : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
    print("elapsed time : %2f"%(time.time()-t))
    print("executaion_time_cost :",executaion_time_cost/(l))
    with open(result_path, 'a') as f:
        f.write("Central_DPG : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
        f.write("Elapsed time : %2f \n"%(time.time()-t))
        f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l)))


elif choice==2: # Dist
    Agents_list=[]
    for n in range(num_of_agents):
        Agents_list.append(PolicyGradientAgent(lr=1e-3,state_dims=state_num,gamma=0.9,n_actions=action_num))

    for i in range(num_of_agents):
        Agents_list[i].load_model('test/models/DPG/Dist/Distributed_DPG_index_'+str(n)+'.pth')

    
    executaion_time_cost=0
    t=time.time()

    reward_dpg_list = list()               
    for k in range(max_episode):
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                agent=Agents_list[n]

                st=time.time()
                a_agent = agent.choose_action(s_actor_agent) 
                executaion_time_cost=executaion_time_cost+time.time()-st    
                p_agent=env.get_power_set(min_p)[a_agent] 
                a[n*maxM:(n+1)*maxM]=a_agent
                p[n*maxM:(n+1)*maxM]=p_agent
            s_actor_next, _, rate, r,rate_all_agents = env.step(p)
            s_actor = s_actor_next
            reward_dpg_list.append(r)  
    reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
    DPG_mean=np.nanmean(reward_dpg_list)
    DPG_std=np.nanstd(reward_dpg_list)
    print("Distributed_DPG : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
    print("elapsed time : %2f"%(time.time()-t))
    print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
    with open(result_path, 'a') as f:
        f.write("Distributed__DPG : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
        f.write("Elapsed time : %2f \n"%(time.time()-t))
        f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l*num_of_agents)))


elif choice==3: # FL
    method="Ensemble_avg_"
    file_name=method+"FL_DPG_Fed_AggPer"
    #AggPerNum=['1','10','100','1000']
    AggPerNum=['100']

    N=1
    
    for agg in range(4):
        executaion_time_cost=0
        t=time.time()

        DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        agent=DPG_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    st=time.time()
                    a_agent = agent.choose_action(s_actor_agent) 
                    executaion_time_cost=executaion_time_cost+time.time()-st    
                    p_agent=env.get_power_set(min_p)[a_agent] 
                    a[n*maxM:(n+1)*maxM]=a_agent
                    p[n*maxM:(n+1)*maxM]=p_agent
                s_actor_next, _, rate, r,rate_all_agents = env.step(p)
                s_actor = s_actor_next
                reward_dpg_list.append(r)  
        reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
        DPG_mean=np.nanmean(reward_dpg_list)
        DPG_std=np.nanstd(reward_dpg_list)
        print(file_name+'_'+AggPerNum[agg]+"_(1) : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write(file_name+'_'+AggPerNum[agg]+"_("+str(N)+") : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")


elif choice==4:
    folder='Personal_Ensemble_avg_'

    # 4
    AggPer=['100']
    
    # 4
    # P=[0.1,0.2,0.5,0.8]
    # G=[0.9,0.8,0.5,0.2]

    P=[0.5]
    G=[0.5]

    N=1

    Personal_DPG_agent=PersonalPG(lr=1e-3,state_dims=state_num,gamma=0.9,n_actions=action_num)
    Agents_list=[]

    for n in range(num_of_agents):
        Agents_list.append(PersonalPG(lr=1e-3,state_dims=state_num,gamma=0.9,n_actions=action_num))


    for AggPerNum in range(len(AggPer)):
        for PG in range(len(P)):
            

            for i in range(num_of_agents):
                Agents_list[i].load_model('test/models/DPG/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_DPG_Fed_AggPer_'+str(AggPer[AggPerNum])+'__P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')_index_'+str(n)+'.pth')

            Personal_DPG_agent.load_model('test/models/DPG/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_DPG_Fed_AggPer_'+str(AggPer[AggPerNum])+'__P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+').pth')
            
            
            executaion_time_cost=0
            t=time.time()

            reward_dpg_list = list()                
            for k in range(max_episode):
                s_actor, _ = env.reset()
                for i in range(int(Ns)-1):
                    for n in range(num_of_agents):
                        s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                        agent=Agents_list[n]

                        st=time.time()

                        #a_agent = agent.select_action(s_actor_agent,k) # 
                        P_prob = agent.choose_action(s_actor_agent) 
                        G_prob = Personal_DPG_agent.choose_action(s_actor_agent) 
                        a_agent =  agent.add_mix_action(P_prob*P[PG]+G_prob*G[PG])
                        

                        a_agent=np.around(a_agent).astype(int)

                        executaion_time_cost=executaion_time_cost+time.time()-st       
                        p_agent=env.get_power_set(min_p)[a_agent]       
                        a[n*maxM:(n+1)*maxM]=a_agent
                        p[n*maxM:(n+1)*maxM]=p_agent
                    s_actor_next, _, rate, r,rate_all_agents = env.step(p)
                    s_actor = s_actor_next
                    reward_dpg_list.append(r) 
            reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
            DPG_mean=np.nanmean(reward_dpg_list)
            DPG_std=np.nanstd(reward_dpg_list)
            print(folder+"_FL_DPG_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_(1) : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
            print("elapsed time : %2f"%(time.time()-t))
            print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
            with open(result_path, 'a') as f:
                f.write(folder+"_FL_DPG_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_(1) : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
                f.write("Elapsed time : %2f \n"%(time.time()-t))
                f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
        with open(result_path, 'a') as f:
            f.write("\n")
    with open(result_path, 'a') as f:
        f.write("\n\n")



