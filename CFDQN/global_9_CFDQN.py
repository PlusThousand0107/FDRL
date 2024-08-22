#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:31:30 2020

@author: peymantehrani
"""


import numpy as np
from DQN_agent_pytorch import DQNAgent
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
import torch

import argparse # add parse

# def parse_args():
#     parse=argparse.ArgumentParser()
#     parse.add_argument('AggPer',type=int)
#     parse.add_argument('Num',type=int)
#     args=parse.parse_args()
#     return args


if __name__ == '__main__':
    fd = 10 
    Ts = 20e-3 
    n_x = 5 
    n_y = 5 
    L = 2
    C = 16
    maxM = 4   # user number in one BS
    min_dis = 0.01 #km
    max_dis = 1. #km 1.
    max_p = 38. #dBm
    min_p = 5
    p_n = -114. #dBm
    power_num = 10  #action_num 10
    seed=11
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    batch_size = 500
    max_episode = 7000 
    buffer_size = 50000
    Ns = 11
    
    
    env.set_Ns(Ns) 
    INITIAL_EPSILON = 0.2 
    FINAL_EPSILON = 0.0001

    state_num = env.state_num
    action_num = env.power_num 
    
    num_of_agents=n_x*n_y
    Agents_list=[]
    reward_lists_of_list=[]
    mean_reward_lists_of_list=[]
    

    Global_agents_list=[]
    global_agent_G = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_1 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_2 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_3 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_4 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_5 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_6 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_7 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_8 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    global_agent_9 = DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000) 
    
    Global_agents_list.append(global_agent_1)
    Global_agents_list.append(global_agent_2)
    Global_agents_list.append(global_agent_3)
    Global_agents_list.append(global_agent_4)
    Global_agents_list.append(global_agent_5)
    Global_agents_list.append(global_agent_6)
    Global_agents_list.append(global_agent_7)
    Global_agents_list.append(global_agent_8)
    Global_agents_list.append(global_agent_9)

    for n in range(num_of_agents):
        Agents_list.append(DQNAgent( gamma=0, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.2, FINAL_EPSILON=0.0001,max_episode=5000,
                 replace=1000)) 
        reward_lists_of_list.append([])
        mean_reward_lists_of_list.append([])


    interval = 100

    # args=parse_args()
    # AggPer=args.AggPer
    # Num=args.Num
    AggPer=10
    Num=1
  
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    a=np.zeros(100).astype(np.int64)
    p=np.zeros(100)
    agent_rewards=np.zeros((num_of_agents,max_episode))

    file_name="Global_Aggr_9Cluster_FL_DQN_Fed_AggPer_"+str(AggPer)+"_("+str(Num)+")"

    n1=[0, 3, 15, 18] 
    n2=[1, 4, 16, 19] 
    n3=[5, 8, 20, 23] 
    n4=[6, 9, 21, 24] 
    n5=[2, 17] 
    n6=[7, 22] 
    n7=[10, 13] 
    n8=[11, 14] 
    n9=[12] 

    for k in range(max_episode):
        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
                agent=Agents_list[n]
                a_agent = agent.select_action(s_actor_agent,k)
                
                p_agent=env.get_power_set(min_p)[a_agent]
                
                a[n*maxM:(n+1)*maxM]=a_agent
                p[n*maxM:(n+1)*maxM]=p_agent

                
            s_actor_next, _, rate, r,rate_all_agents = env.step(p)
            
            for n in range(num_of_agents):
                agent=Agents_list[n]
                agent.store_transition(s_actor[n*maxM:(n+1)*maxM,:], a[n*maxM:(n+1)*maxM]
                                       , rate[n*maxM:(n+1)*maxM],s_actor_next[n*maxM:(n+1)*maxM,:])
                reward_lists_of_list[n].append(np.mean(rate_all_agents[n*maxM:(n+1)*maxM]))
                
            s_actor = s_actor_next
            reward_dqn_list.append(r)
            all_reward.append(r)
            
        for n in range(num_of_agents):
            agent=Agents_list[n]
            agent.learn()
            agent_rewards[n,k] =np.mean(reward_lists_of_list[n][-(Ns-1):])
            
            

        if k % AggPer == 0:
            # 1
            global_dict_1 = global_agent_1.q_eval.state_dict()

            for kd in global_dict_1.keys():
                global_dict_1[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n1], 0).mean(0)

            global_agent_1.q_eval.load_state_dict(global_dict_1)


            # 2
            global_dict_2 = global_agent_2.q_eval.state_dict()

            for kd in global_dict_2.keys():
                global_dict_2[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n2], 0).mean(0)

            global_agent_2.q_eval.load_state_dict(global_dict_2)

            # 3
            global_dict_3 = global_agent_3.q_eval.state_dict()

            for kd in global_dict_3.keys():
                global_dict_3[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n3], 0).mean(0)

            global_agent_3.q_eval.load_state_dict(global_dict_3)

            # 4
            global_dict_4 = global_agent_4.q_eval.state_dict()

            for kd in global_dict_4.keys():
                global_dict_4[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n4], 0).mean(0)

            global_agent_4.q_eval.load_state_dict(global_dict_4)

            # 5
            global_dict_5 = global_agent_5.q_eval.state_dict()

            for kd in global_dict_5.keys():
                global_dict_5[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n5], 0).mean(0)

            global_agent_5.q_eval.load_state_dict(global_dict_5)

            # 6
            global_dict_6 = global_agent_6.q_eval.state_dict()

            for kd in global_dict_6.keys():
                global_dict_6[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n6], 0).mean(0)

            global_agent_6.q_eval.load_state_dict(global_dict_6)

            # 7
            global_dict_7 = global_agent_7.q_eval.state_dict()

            for kd in global_dict_7.keys():
                global_dict_7[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n7], 0).mean(0)

            global_agent_7.q_eval.load_state_dict(global_dict_7)

            # 8
            global_dict_8 = global_agent_8.q_eval.state_dict()

            for kd in global_dict_8.keys():
                global_dict_8[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n8], 0).mean(0)

            global_agent_8.q_eval.load_state_dict(global_dict_8)

            # 9
            global_dict_9 = global_agent_9.q_eval.state_dict()

            for kd in global_dict_9.keys():
                global_dict_9[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in n9], 0).mean(0)

            global_agent_9.q_eval.load_state_dict(global_dict_9)


            # G
            global_dict_G = global_agent_G.q_eval.state_dict()

            for kd in global_dict_G.keys():
                global_dict_G[kd] = torch.stack([Agents_list[n].q_eval.state_dict()[kd] for n in range(9)], 0).mean(0)


            global_agent_G.q_eval.load_state_dict(global_dict_G)

            for n in range(num_of_agents):
                Agents_list[n].q_eval.load_state_dict(global_agent_G.q_eval.state_dict())

            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 

            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            for n in range(num_of_agents):
                mean_reward_lists_of_list[n].append(np.mean(agent_rewards[n,-interval:]))
            if reward > max_reward:

                global_agent_G.save_models('models_param/CFDQN/'+file_name+'.pth')
                #global_agent_G.save_models("models_param/PPO/"+file_name+"/"+file_name+"_global_agentG__("+str(cnt)+")")

                max_reward = reward
            print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()



print(file_name+" : "+str(max_reward))
np.save('npfiles/CFDQN/mean_reward/'+file_name+'__mean_reward.npy',np.array(mean_reward))
np.save('npfiles/CFDQN/reward/'+file_name+'__reward.npy',np.array(reward_hist))


plt.plot(reward_hist)
plt.savefig('figs/CFDQN/reward/'+file_name+'__reward.png')
#plt.show()
plt.close()

plt.plot(mean_reward)
plt.savefig('figs/CFDQN/mean_reward/'+file_name+'__mean_reward.png')
#plt.show()
plt.close()
