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
n_x = 7
n_y = 7
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

choice=24
# Central 1
# Dist 2
# FL 3
# Personal 4

# Layer 6

# Adj 9
# Adj Norm 10
# Neighbor 11
# Neighbor Norm 12
# Layer 13
# Layer Norm 14
# Layer OneHot 15
# L3 Norm 16
# Corner Norm 17
# 2 Cluster 18
# 2 Cluster Neighbor Norm 19
# 3 Cluster 20
# 3 Cluster AllMean 21
# Global + 2 Cluster 22
# Center Concer 2 Cluster 23
# Global Aggr 2 Cluster 24
# Global Aggr 9 Cluster 25

############################################################################################################################




if choice==1: # Central
    PPO_mean=[]
    num=7
    file_name="Actor256X128X64_StateAvg_Cent_PPO___Epochs_5_Eps_0.05_Batch_512_(1)"

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
    file_name="Dist_PPO___Epochs_5_Eps_0.1_Batch_32_(3)"



    Agents_list=[]
    for n in range(num_of_agents):
        Agents_list.append(PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num))

    for i in range(num_of_agents):
        Agents_list[i].load_model('test/models/PPO/'+file_name+'/'+file_name+'_index_'+str(i)+'__'+str(n)+'_actor.pth')
        # Dist_PPO___Epochs_5_Eps_0.1_Batch_32_(3)_index_1__(0)_actor
        # Dist_PPO___Epochs_5_Eps_0.1_Batch_32_(3)_index_6__(0)_critic
        #PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')

    
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
    num=15
    file_name="Entropy005_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
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
    num=10
    file_name="EnsembleAvg_Personal_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32__P0.5_G0.5_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
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


elif choice==40:
    folder='Personal2'
    '''
    # 4
    AggPer=['10']
    
    # 4
    # P=[0.1,0.2,0.5,0.8]
    # G=[0.9,0.8,0.5,0.2]

    P=[0.5]
    G=[0.5]

    #args=parse_args()
    #N=args.N
    N=1

    Personal_PPO_agent=PersonalPPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    Agents_list=[]

    for n in range(num_of_agents):
        Agents_list.append(PersonalPPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num))


    for AggPerNum in range(len(AggPer)):
        for PG in range(len(P)):
            

            for i in range(num_of_agents):
                Agents_list[i].load_model('test/models/PPO/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_PPO___Epochs_10_Eps_0.1__P0.5_G0.5_(1)_index_'+str(i)+'_actor.pth')

            #Personal_PPO_agent.load_model('test/models/PPO/'+folder+'/AggPer'+str(AggPer[AggPerNum])+'_P'+str(P[PG])+'_G'+str(G[PG])+'_('+str(N)+')/'+folder+'_FL_PPO___Epochs_10_Eps_0.1__P0.5_G0.5_(1)_actor.pth')
            
            
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

                        #a_agent = agent.select_action(s_actor_agent,k) # 

                        P_prob = agent.choose_action(s_actor_agent) #
                        G_prob = Personal_PPO_agent.choose_action(s_actor_agent) #
                        a_agent =  agent.add_mix_action(P_prob*P[PG]+G_prob*G[PG]) #

                        a_agent=np.around(a_agent).astype(int) #

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
            print(folder+"_FL_DPG_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_("+str(N)+") : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
            print("elapsed time : %2f"%(time.time()-t))
            print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
            with open(result_path, 'a') as f:
                f.write(folder+"_FL_DPG_Fed_AggPer_"+str(AggPer[AggPerNum])+"__P"+str(P[PG])+"_G"+str(G[PG])+"_("+str(N)+") : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
                f.write("Elapsed time : %2f \n"%(time.time()-t))
                f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
        with open(result_path, 'a') as f:
            f.write("\n")
    with open(result_path, 'a') as f:
        f.write("\n\n")
    '''


elif choice==5: # 
    # method="Ensemble_avg_beta05_"
    # file_name=method+"FL_DPG_Fed_AggPer"
    # #AggPerNum=['1','10','100','1000']
    # AggPerNum=['100']

    # N=1
    
    for agg in range(1):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/Adj_FL_PPO___Epochs_5_Eps_0.05_AggPer_100_(1)_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()         

        n2=[0,20]
        n3=[4,9,10,19,24]
        n4=[1,2,3,21,22,23]
        n5=[5,14,15]
        n6=[6,7,8,11,12,13,16,17,18]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                    if n in n2:
                        local_index=[[2],[2],[2],[2]]
                    elif n in n5:
                        local_index=[[5],[5],[5],[5]]
                    elif n in n3:
                        local_index=[[3],[3],[3],[3]]
                    elif n in n4:
                        local_index=[[4],[4],[4],[4]]
                    else: local_index=[[6],[6],[6],[6]]

                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)
                    


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
        print("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")





elif choice==6: # 
    # method="Ensemble_avg_beta05_"
    # file_name=method+"FL_DPG_Fed_AggPer"
    # #AggPerNum=['1','10','100','1000']
    # AggPerNum=['100']

    # N=1
    
    for agg in range(1):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/Layer_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_batch_32_(1)_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()         

        l1=[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24]
        l2=[6,7,8,11,13,16,17,18]
        l3=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in l3:
                        local_index=[[3],[3],[3],[3]]
                    elif n in l2:
                        local_index=[[2],[2],[2],[2]]
                    else: local_index=[[1],[1],[1],[1]]


                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)
                    


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
        print("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")




elif choice==7: # 
    # method="Ensemble_avg_beta05_"
    # file_name=method+"FL_DPG_Fed_AggPer"
    # #AggPerNum=['1','10','100','1000']
    # AggPerNum=['100']

    # N=1
    
    for agg in range(1):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/Neighbor_FL_PPO___Epochs_10_Eps_0.05_AggPer_10_batch_16_(1)_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()         

        n7=[0,20]
        n8=[4,24]
        n9=[9,19]
        n10=[1,21]
        n11=[3,5,10,15,23]
        n12=[2,22]
        n13=[8,14,18]
        n15=[6,16]
        n16=[7,11,17]
        n18=[13]
        n19=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n11:
                        local_index=[[11],[11],[11],[11]]
                    elif n in n13:
                        local_index=[[13],[13],[13],[13]]
                    elif n in n16:
                        local_index=[[16],[16],[16],[16]]
                    elif n in n7:
                        local_index=[[7],[7],[7],[7]]
                    elif n in n8:
                        local_index=[[8],[8],[8],[8]]
                    elif n in n9:
                        local_index=[[9],[9],[9],[9]]
                    elif n in n10:
                        local_index=[[10],[10],[10],[10]]
                    elif n in n12:
                        local_index=[[12],[12],[12],[12]]
                    elif n in n15:
                        local_index=[[15],[15],[15],[15]]
                    elif n in n18:
                        local_index=[[18],[18],[18],[18]]
                    else : local_index=[[19],[19],[19],[19]]


                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)
                    


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
        print("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")




elif choice==8: # 
    # method="Ensemble_avg_beta05_"
    # file_name=method+"FL_DPG_Fed_AggPer"
    # #AggPerNum=['1','10','100','1000']
    # AggPerNum=['100']

    # N=1
    
    for agg in range(1):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/Neighbor_Norm_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_batch_16_(1)_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()         

        n7=[0,20]
        n8=[4,24]
        n9=[9,19]
        n10=[1,21]
        n11=[3,5,10,15,23]
        n12=[2,22]
        n13=[8,14,18]
        n15=[6,16]
        n16=[7,11,17]
        n18=[13]
        n19=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n11:
                        local_index=[[0.333],[0.333],[0.333],[0.333]]
                    elif n in n13:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    elif n in n16:
                        local_index=[[0.75],[0.75],[0.75],[0.75]]
                    elif n in n7:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n8:
                        local_index=[[0.083],[0.083],[0.083],[0.083]]
                    elif n in n9:
                        local_index=[[0.166],[0.166],[0.166],[0.166]]
                    elif n in n10:
                        local_index=[[0.25],[0.25],[0.25],[0.25]]
                    elif n in n12:
                        local_index=[[0.416],[0.416],[0.416],[0.416]]
                    elif n in n15:
                        local_index=[[0.666],[0.666],[0.666],[0.666]]
                    elif n in n18:
                        local_index=[[0.916],[0.916],[0.916],[0.916]]
                    else : local_index=[[1.0],[1.0],[1.0],[1.0]]

                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)
                    


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
        print("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f "%(DPG_mean,DPG_std))
        print("elapsed time : %2f"%(time.time()-t))
        print("executaion_time_cost :",executaion_time_cost/(l*num_of_agents))
        with open(result_path, 'a') as f:
            f.write("FL_PPO___Epochs_5_Eps_0.05_(1)_actor : Mean = %.4f , STD = %.4f \n"%(DPG_mean,DPG_std))
            f.write("Elapsed time : %2f \n"%(time.time()-t))
            f.write("Executaion_time_cost : %2f \n"%(executaion_time_cost/(l*num_of_agents)))
    with open(result_path, 'a') as f:
        f.write("\n\n")

elif choice==9: # FL


    # N=1
    PPO_mean=[]
    num=1
    file_name="Adj_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              

        n2=[0,20]
        n3=[4,9,10,19,24]
        n4=[1,2,3,21,22,23]
        n5=[5,14,15]
        n6=[6,7,8,11,12,13,16,17,18]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n2:
                        local_index=[[2],[2],[2],[2]]
                    elif n in n5:
                        local_index=[[5],[5],[5],[5]]
                    elif n in n3:
                        local_index=[[3],[3],[3],[3]]
                    elif n in n4:
                        local_index=[[4],[4],[4],[4]]
                    else: local_index=[[6],[6],[6],[6]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==10: # FL


    # N=1
    PPO_mean=[]
    num=26
    file_name="Adj_Norm_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              

        n2=[0,20]
        n3=[4,9,10,19,24]
        n4=[1,2,3,21,22,23]
        n5=[5,14,15]
        n6=[6,7,8,11,12,13,16,17,18]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n2:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n5:
                        local_index=[[0.75],[0.75],[0.75],[0.75]]
                    elif n in n3:
                        local_index=[[0.25],[0.25],[0.25],[0.25]]
                    elif n in n4:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    else: local_index=[[1.0],[1.0],[1.0],[1.0]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==11: # FL


    # N=1
    PPO_mean=[]
    num=2
    file_name="Neighbor_FL_PPO___Epochs_5_Eps_0.1_AggPer_100_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              

        n7=[0,20]
        n8=[4,24]
        n9=[9,19]
        n10=[1,21]
        n11=[3,5,10,15,23]
        n12=[2,22]
        n13=[8,14,18]
        n15=[6,16]
        n16=[7,11,17]
        n18=[13]
        n19=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]



                    if n in n11:
                        local_index=[[11],[11],[11],[11]]
                    elif n in n13:
                        local_index=[[13],[13],[13],[13]]
                    elif n in n16:
                        local_index=[[16],[16],[16],[16]]
                    elif n in n7:
                        local_index=[[7],[7],[7],[7]]
                    elif n in n8:
                        local_index=[[8],[8],[8],[8]]
                    elif n in n9:
                        local_index=[[9],[9],[9],[9]]
                    elif n in n10:
                        local_index=[[10],[10],[10],[10]]
                    elif n in n12:
                        local_index=[[12],[12],[12],[12]]
                    elif n in n15:
                        local_index=[[15],[15],[15],[15]]
                    elif n in n18:
                        local_index=[[18],[18],[18],[18]]
                    else : local_index=[[19],[19],[19],[19]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==12: # FL


    # N=1
    PPO_mean=[]
    num=23
    file_name="EnsembleNeighbor_Neighbor_Personal_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32__P0.5_G0.5_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              

        n7=[0,20]
        n8=[4,24]
        n9=[9,19]
        n10=[1,21]
        n11=[3,5,10,15,23]
        n12=[2,22]
        n13=[8,14,18]
        n15=[6,16]
        n16=[7,11,17]
        n18=[13]
        n19=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]



                    if n in n11:
                        local_index=[[0.333],[0.333],[0.333],[0.333]]
                    elif n in n13:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    elif n in n16:
                        local_index=[[0.75],[0.75],[0.75],[0.75]]
                    elif n in n7:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n8:
                        local_index=[[0.083],[0.083],[0.083],[0.083]]
                    elif n in n9:
                        local_index=[[0.166],[0.166],[0.166],[0.166]]
                    elif n in n10:
                        local_index=[[0.25],[0.25],[0.25],[0.25]]
                    elif n in n12:
                        local_index=[[0.416],[0.416],[0.416],[0.416]]
                    elif n in n15:
                        local_index=[[0.666],[0.666],[0.666],[0.666]]
                    elif n in n18:
                        local_index=[[0.916],[0.916],[0.916],[0.916]]
                    else : local_index=[[1.0],[1.0],[1.0],[1.0]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==13: # FL


    # N=1
    PPO_mean=[]
    num=1
    file_name="Layer_FL_PPO___Epochs_5_Eps_0.05_AggPer_100_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              


        l1=[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24]
        l2=[6,7,8,11,13,16,17,18]
        l3=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]



                    if n in l3:
                        local_index=[[3],[3],[3],[3]]
                    elif n in l2:
                        local_index=[[2],[2],[2],[2]]
                    else: local_index=[[1],[1],[1],[1]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==14: # FL


    # N=1
    PPO_mean=[]
    num=6
    file_name="Layer_Norm_FL_PPO___Epochs_5_Eps_0.05_AggPer_100_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              


        l1=[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24]
        l2=[6,7,8,11,13,16,17,18]
        l3=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in l3:
                        local_index=[[1.0],[1.0],[1.0],[1.0]]
                    elif n in l2:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    else: local_index=[[0.0],[0.0],[0.0],[0.0]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==15: # FL


    # N=1
    PPO_mean=[]
    num=27
    file_name="Layer_OneHot_FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              


        l1=[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24]
        l2=[6,7,8,11,13,16,17,18]
        l3=[12]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in l3:
                        local_index=[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]
                    elif n in l2:
                        local_index=[[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
                    else: local_index=[[1,0,0],[1,0,0],[1,0,0],[1,0,0]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==16: # FL


    # N=1
    PPO_mean=[]
    num=21
    file_name="L3_Norm_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              


        n1=[0,4,9,19,20,24]
        n2=[1,2,3,5,8,10,14,15,18,21,22,23]
        n3=[6,7,11,12,13,16,17]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n1:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n3:
                        local_index=[[1.0],[1.0],[1.0],[1.0]]
                    else:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==17: # FL


    # N=1
    PPO_mean=[]
    num=16
    file_name="Corner_Norm_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'__('+str(i)+')_actor.pth')
        agent=PPO_agent
        reward_dpg_list = list()              



        n1=[0,4,20,24]
        n2=[1,2,3,5,8,9,10,14,15,18,19,21,22,23]
        n3=[6,7,11,12,13,16,17]

        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):
                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    if n in n1:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n3:
                        local_index=[[1.0],[1.0],[1.0],[1.0]]
                    else:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==18: # FL


    # N=1
    PPO_mean=[]
    num=14
    file_name="test/models/PPO/2Cluster_FL_PPO___Epochs_5_Eps_0.05_AggPer_100_Batch_32_(1)"
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24] # 16
    n2=[6, 7, 8, 11, 12, 13, 16, 17, 18] # 9

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    else: agent=PPO_agent_2

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


elif choice==19: # FL


    # N=1
    PPO_mean=[]
    num=8
    file_name="2Cluster_Neighbor_Norm_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24] # 16
    n2=[6, 7, 8, 11, 12, 13, 16, 17, 18] # 9

    n7=[0,20]
    n8=[4,24]
    n9=[9,19]
    n10=[1,21]
    n11=[3,5,10,15,23]
    n12=[2,22]
    n13=[8,14,18]
    n15=[6,16]
    n16=[7,11,17]
    n18=[13]
    n19=[12]

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    else: agent=PPO_agent_2

                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                    if n in n11:
                        local_index=[[0.333],[0.333],[0.333],[0.333]]
                    elif n in n13:
                        local_index=[[0.5],[0.5],[0.5],[0.5]]
                    elif n in n16:
                        local_index=[[0.75],[0.75],[0.75],[0.75]]
                    elif n in n7:
                        local_index=[[0.0],[0.0],[0.0],[0.0]]
                    elif n in n8:
                        local_index=[[0.083],[0.083],[0.083],[0.083]]
                    elif n in n9:
                        local_index=[[0.166],[0.166],[0.166],[0.166]]
                    elif n in n10:
                        local_index=[[0.25],[0.25],[0.25],[0.25]]
                    elif n in n12:
                        local_index=[[0.416],[0.416],[0.416],[0.416]]
                    elif n in n15:
                        local_index=[[0.666],[0.666],[0.666],[0.666]]
                    elif n in n18:
                        local_index=[[0.916],[0.916],[0.916],[0.916]]
                    else : local_index=[[1.0],[1.0],[1.0],[1.0]]
                
                    s_actor_agent=np.append(s_actor_agent,local_index,axis=1)


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


elif choice==20: # FL


    # N=1
    PPO_mean=[]
    num=5
    file_name="Critic_Global_3Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_3 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0,4,20,24] # 16
    n2=[1,2,3,5,8,9,10,14,15,18,19,21,22,23] # 9
    n3=[6,7,11,12,13,16,17]

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        PPO_agent_3.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent3__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    elif n in n2:
                        agent=PPO_agent_2
                    else: agent=PPO_agent_3

                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]


                    st=time.time()
                    a_agent = agent.choose_action(s_actor_agent) 
                    #a_agent[a_agent == 9]=8
                    executaion_time_cost=executaion_time_cost+time.time()-st    
                    p_agent=env.get_power_set(min_p)[a_agent] 
                    a[n*maxM:(n+1)*maxM]=a_agent
                    p[n*maxM:(n+1)*maxM]=p_agent
                s_actor_next, _, rate, r,rate_all_agents = env.step(p)
                s_actor = s_actor_next
                reward_dpg_list.append(r)  
                #print(a)
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



elif choice==21: # FL


    # N=1
    PPO_mean=[]
    num=6
    file_name="3Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_EnvAllMean_(1)"
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_3 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0,4,20,24] # 16
    n2=[1,2,3,5,8,9,10,14,15,18,19,21,22,23] # 9
    n3=[6,7,11,12,13,16,17]

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        PPO_agent_3.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent3__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    elif n in n2:
                        agent=PPO_agent_2
                    else: agent=PPO_agent_3

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



elif choice==22: # FL


    # N=1
    PPO_mean=[]
    num=15
    file_name="Global_2Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_P5G5_(1)"

    PPO_agent_G = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24] # 16
    n2=[6, 7, 8, 11, 12, 13, 16, 17, 18] # 9

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_G.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global__('+str(i)+')_actor.pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    else: agent=PPO_agent_2

                    s_actor_agent=s_actor[n*maxM:(n+1)*maxM]

                    st=time.time()

                    #
                    a_agent = agent.choose_action(s_actor_agent) * 0.5 + PPO_agent_G.choose_action(s_actor_agent) * 0.5
                    a_agent = np.round(a_agent).astype(int)
                    #a_agent[a_agent != 9]=0
                    #

                    executaion_time_cost=executaion_time_cost+time.time()-st    
                    p_agent=env.get_power_set(min_p)[a_agent] 
                    a[n*maxM:(n+1)*maxM]=a_agent
                    p[n*maxM:(n+1)*maxM]=p_agent
                s_actor_next, _, rate, r,rate_all_agents = env.step(p)
                s_actor = s_actor_next
                reward_dpg_list.append(r)  
                #print(a)
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


elif choice==23: # FL


    # N=1
    PPO_mean=[]
    num=12
    file_name="Center_Corner_2Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    PPO_agent_1 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)
    PPO_agent_2 = PPOAgent(epochs=5,eps=0.05,state_dims=state_num,gamma=0.9,n_actions=action_num)

    n1=[0, 4, 12, 20, 24] # 16
    n2=[1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23] # 9

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent_1.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent1__('+str(i)+')_actor.pth')
        PPO_agent_2.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agent2__('+str(i)+')_actor.pth')
        #agent=PPO_agent
        reward_dpg_list = list()                 
        for k in range(max_episode):
            s_actor, _ = env.reset()
            for i in range(int(Ns)-1):
                for n in range(num_of_agents):

                    if n in n1:
                        agent=PPO_agent_1
                    else: agent=PPO_agent_2

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


elif choice==24: # FL


    # N=1
    PPO_mean=[]
    num=9
    file_name="7X7_Global_Aggr_9Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agentG__('+str(i)+')_actor.pth')
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



elif choice==25: # FL


    # N=1
    PPO_mean=[]
    num=19
    file_name="StateAvg_Global_Aggr_2Cluster_FL_PPO___Epochs_5_Eps_0.15_AggPer_10_Batch_32_(1)"
    

    for i in range(num):
        executaion_time_cost=0
        t=time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        #DPG_agent.load_model('models_param/DPG/'+file_name+'_'+AggPerNum[agg]+'_('+str(N)+').pth')
        PPO_agent.load_model('test/models/PPO/'+file_name+'/'+file_name+'_global_agentG__('+str(i)+')_actor.pth')
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