#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment_CU import Env_cellular
from DQN_agent_pytorch import DQNAgent
#from Personal_DQN_agent_pytorch import DQNAgent
import time
import torch

import argparse # add parse

def parse_args():
    parse=argparse.ArgumentParser()
    parse.add_argument('N',type=int)
#    parse.add_argument('Num',type=int)
    args=parse.parse_args()
    return args


fd = 10
Ts = 20e-3
n_x = 9
n_y = 9
L = 2
C = 16
maxM = 4   # user number in one BS
min_dis = 0.01 #km
max_dis = 1. #km
max_p = 38. #dBm
min_p = 5
p_n = -114. #dBm
power_num = 10 #action_num

# args=parse_args()
# N=args.N
seed=91

Ns=21 # 11
max_episode=100 #for test

env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env.set_Ns(Ns)

batch_size = 512
buffer_size = 50000
INITIAL_EPSILON = 0.2 
FINAL_EPSILON = 0.0001
state_num = env.state_num
action_num = env.power_num  
num_of_agents=n_x*n_y
a=np.zeros(n_x*n_y*maxM).astype(np.int64)
p=np.zeros(n_x*n_y*maxM)

dqn_agent=DQNAgent( gamma=0.9, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0, FINAL_EPSILON=0,max_episode=5000,
                 replace=1000)

l=max_episode*(Ns-1)



result_path = 'test/result/DQN/DQN.txt'

choice=4
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
    method="Ensemble_avg_beta05_replace500_init05_end4500_seed91"
    file_name=method+"_FL_DQN_Fed_AggPer"
    #AggPerNum=['1','10','100','1000']
    AggPerNum=['100']


    args=parse_args()
    N=args.N
    #N=1
    

    for agg in range(4):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        dqn_agent.load_models('models_param/DQN/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
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



    





elif choice==4:
    folder='9X9_Ensemble_Personal_beta05'

    # 4
    #AggPer=['1','10','100','1000']
    AggPer=['100']


    # 4
    #P=[0.1,0.2,0.5,0.8]
    #G=[0.9,0.8,0.5,0.2]
    #P=[0.1]
    #G=[0.9]
    P=[0.5]
    G=[0.5]


    #Num=args.Num


    #args=parse_args()
    #N=args.N
    N=1
    cnt=1

    Agents_list=[]

    for n in range(num_of_agents):
        Agents_list.append(DQNAgent( gamma=0.9, lr=1e-3, n_actions=action_num, state_dim=state_num,
                buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.0, FINAL_EPSILON=0.0,max_episode=5000,
                replace=1000)) # gamma=0




    for AggPerNum in range(len(AggPer)):
        for PG in range(len(P)):

            for i in range(num_of_agents):
                Agents_list[i].load_models('test/models/Personal_DQN/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_DQN_Fed_AggPer_'+str(AggPer[AggPerNum])+'__P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')_index_'+str(n)+"__("+str(cnt)+")"+'.pth')

            dqn_agent.load_models('test/models/Personal_DQN/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_DQN_Fed_AggPer_'+str(AggPer[AggPerNum])+'__P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')'+"__("+str(cnt)+")"+'.pth')
            
            
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


                        a_agent = agent.select_action(s_actor_agent,k) * P[PG] + dqn_agent.select_action(s_actor_agent,k) * G[PG]


                        
                        a_agent=np.around(a_agent).astype(int)

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
            print(folder+"_FL_DQN_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_("+str(N)+") : Mean = %.4f , STD = %.4f "%(DQN_dist_mean,DQN_dist_std))
            print("elapsed time : %2f"%(time.time()-t))
            print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
            with open(result_path, 'a') as f:
                f.write(folder+"_FL_DQN_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_("+str(N)+") : Mean = %.4f , STD = %.4f \n"%(DQN_dist_mean,DQN_dist_std))
                f.write("Elapsed time : %2f \n"%(time.time()-t))
                f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
        with open(result_path, 'a') as f:
            f.write("\n")
    with open(result_path, 'a') as f:
        f.write("\n\n")



