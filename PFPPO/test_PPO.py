#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Environment_CU import Env_cellular
# from Reinforce_Pytorch import PolicyGradientAgent
# from Personal_Reinforce_Pytorch import PolicyGradientAgent as PersonalPG
from PPO_agent import PPOAgent
#from Personal_PPO_agent import PPOAgent as PersonalPPOAgent
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

# args=parse_args()
# N=args.N
seed=11


Ns=21 # 11
max_episode=100 #for test

env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env.set_Ns(Ns)

batch_size = 512
buffer_size = 50000
# INITIAL_EPSILON = 0.2 
# FINAL_EPSILON = 0.0001
state_num = env.state_num
action_num = env.power_num  
num_of_agents=n_x*n_y
a=np.zeros(n_x*n_y*maxM).astype(np.int64)
p=np.zeros(n_x*n_y*maxM)

#DPG_agent=PolicyGradientAgent(lr=1e-3,state_dims=state_num,gamma=0.9,n_actions=action_num)
PPO_agent = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num) # alr=0.0003, clr=0.001

l=max_episode*(Ns-1)


result_path = 'test/result/PPO/PPO.txt'

choice=4
# Central 1
# Dist 2
# FL 3
# Personal 4


############################################################################################################################




if choice==1: # Central
    PPO_mean=[]
    num=4
    file_name="Cent_PPO___Epochs_5_Eps_0.15_Batch_512_(1)"

    for i in range(num):
        
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        #PPO_agent.load_model('test/models/PPO/'+file_name+'/35000_StateAvg_Cent_PPO___Epochs_5_Eps_0.05_Batch_512_(1)__(196)_actor.pth')
        
        agent=PPO_agent
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
            #print(a)
        reward_dpg_list = np.reshape(reward_dpg_list, [max_episode*( Ns-1)])
        DPG_mean=np.nanmean(reward_dpg_list)
        PPO_mean.append(DPG_mean)
        DPG_std=np.nanstd(reward_dpg_list)
        print(file_name+" : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l))
        with open(result_path, 'a') as f:
            f.write(file_name+" : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l)))

    for i in range(num):
        print("- %.4f"%((PPO_mean[i])))
        with open(result_path, 'a') as f:
            f.write("- %.4f\n"%((PPO_mean[i])))
        


elif choice==2: # Dist

    PPO_mean=[]
    num=1
    file_name="Dist_PPO___Epochs_5_Eps_0.05_Batch_32_(4)"



    Agents_list=[]
    for n in range(num_of_agents):
        Agents_list.append(PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num))

    for i in range(num_of_agents):
        Agents_list[i].load_model('test/models/PPO/'+file_name+'/'+file_name+'_index_'+str(i)+'__(0)_actor.pth')


    
    executaion_time_cost=0
    t=time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    print("Distributed_PPO : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
    print("elapsed time : %2f"%(time.time()-t))
    print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
    with open(result_path, 'a') as f:
        f.write("Distributed__PPO : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
        f.write("Elapsed time : %2f \n"%(time.time()-t))
        f.write("Executaion_time_cost : %2f \n\n\n"%(executaion_time_cost/(l*num_of_agents)))

    for i in range(num):
        print("- %.4f"%((PPO_mean[i])))
        with open(result_path, 'a') as f:
            f.write("- %.4f\n"%((PPO_mean[i])))

elif choice==3: # FL


    # N=1
    PPO_mean=[]
    num=1
    file_name="FL_PPO___Epochs_5_Eps_0.2_AggPer_100_Batch_32_(5)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        #PPO_agent.load_model('PPO/test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__(20)_actor.pth')
        agent=PPO_agent
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
        PPO_mean.append(DPG_mean)
        DPG_std=np.nanstd(reward_dpg_list)
        print(file_name+" : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write(file_name+" : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")
    
    for i in range(num):
        print("- %.4f"%((PPO_mean[i])))
        with open(result_path, 'a') as f:
            f.write("- %.4f\n"%((PPO_mean[i])))


elif choice==4: # FL


    # N=1
    PPO_mean=[]
    num=1
    file_name="PFPPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32__P0.5G0.5_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        #PPO_agent.load_model('PFPPO/test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        PPO_agent.load_model('models_param/PFPPO/'+file_name+'/'+file_name+'__(0)_actor.pth')
        agent=PPO_agent
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
        PPO_mean.append(DPG_mean)
        DPG_std=np.nanstd(reward_dpg_list)
        print(file_name+" : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write(file_name+" : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")
    
    for i in range(num):
        print("- %.4f"%((PPO_mean[i])))
        with open(result_path, 'a') as f:
            f.write("- %.4f\n"%((PPO_mean[i])))


