#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Personal_DQN_agent_pytorch import DQNAgent
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
import torch

import argparse # add parse

import os # create directory

# def parse_args():
#     parse=argparse.ArgumentParser()
#     parse.add_argument('AggPer',type=int)
#     parse.add_argument('Personal',type=float)
#     parse.add_argument('Global',type=float)
#     parse.add_argument('Num',type=int)
#     args=parse.parse_args()
#     return args


if __name__ == '__main__':
    fd = 10 # 最大督普勒頻率
    Ts = 20e-3 # 每個時間步長度
    n_x = 5 
    n_y = 5 
    L = 2 # 鄰近區域範圍
    C = 16 # 會被選為做為狀態輸入的用戶數量
    maxM = 4   # user number in one BS
    min_dis = 0.01 #km
    max_dis = 1. #km 1.
    max_p = 38. #dBm
    min_p = 5
    p_n = -114. #dBm
    power_num = 10  #action_num 10
    seed=11 #91
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    batch_size = 512
    max_episode = 7000 
    buffer_size = 50000
    Ns = 11
    

    env.set_Ns(Ns) 
    INITIAL_EPSILON = 0.5 # 0.2
    FINAL_EPSILON = 0.0001
    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num 
    
    num_of_agents=n_x*n_y
    Agents_list=[]
    reward_lists_of_list=[]
    mean_reward_lists_of_list=[]
    
    global_agent = DQNAgent( gamma=0.9, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.5, FINAL_EPSILON=0.0001,max_episode=4500,
                 replace=500) # 0.2 5000

    for n in range(num_of_agents):
        Agents_list.append(DQNAgent( gamma=0.9, lr=1e-3, n_actions=action_num, state_dim=state_num,
                 buffer_size=buffer_size,batch_size=batch_size, INITIAL_EPSILON= 0.5, FINAL_EPSILON=0.0001,max_episode=4500,
                 replace=500)) # 0.2 5000
        reward_lists_of_list.append([])
        mean_reward_lists_of_list.append([])

    interval = 100

    # args=parse_args()
    # AggPer=args.AggPer
    # Personal=args.Personal
    # Global=args.Global
    # Num=args.Num
    AggPer=100
    Personal=0.5 # 個人化比例
    Global=0.5 # 全局化比例
    Num=1
    

    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]
    a=np.zeros(100).astype(np.int64)
    p=np.zeros(100)
    agent_rewards=np.zeros((num_of_agents,max_episode))

    method="Ensemble_Personal_beta05"

    file_name=method+"_FL_DQN_Fed_AggPer_"+str(AggPer)+"__P"+str(Personal)+"_G"+str(Global)+"_("+str(Num)+")"

    directory = 'AggPer'+str(AggPer)+'_P'+str(Personal)+'_G'+str(Global)+"_("+str(Num)+")"
    path = os.path.join('models_param/Personal_DQN/'+method, directory) 
    os.mkdir(path)

    cnt=0

    for k in range(max_episode):
        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM]
                agent=Agents_list[n]

                epsilon  = INITIAL_EPSILON - k * (INITIAL_EPSILON - FINAL_EPSILON ) / max_episode 
                M=4
                rand_idx=np.array(np.random.uniform(size = (M)) < epsilon, dtype = np.int32) # random

                a_agent = agent.select_action(s_actor_agent,rand_idx) * Personal + global_agent.select_action(s_actor_agent,rand_idx) * Global # Personal + Global 混和動作

                a_agent=np.around(a_agent).astype(int) #離散動作空間，須進位至整數

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
            weights=[np.mean(reward_lists_of_list[j][-AggPer:]) for j in range(num_of_agents)]
            
            total_weight=sum(weights)

            global_dict = global_agent.q_eval.state_dict() 
            for kd in global_dict.keys(): 
                weighted_sum = torch.zeros_like(global_dict[kd]) 

                for n in range(num_of_agents):

                    weighted_sum += Agents_list[n].q_eval.state_dict()[kd] * (weights[n]/ total_weight)
        
                global_dict[kd] = weighted_sum 
            global_agent.q_eval.load_state_dict(global_dict) # 更新全局網路
            for n in range(num_of_agents): # 各個 agent 收到更新後的全局模型
                Agents_list[n].q_eval.load_state_dict(global_agent.q_eval.state_dict())
            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            for n in range(num_of_agents):
                mean_reward_lists_of_list[n].append(np.mean(agent_rewards[n,-interval:]))

            if reward > 1.3 and k>6600:
                if reward > max_reward:
                    for n in range(num_of_agents):
                        agent=Agents_list[n]
                        agent.save_models('models_param/Personal_DQN/'+method+'/'+directory+'/'+file_name+'_index_'+str(n)+"__("+str(cnt)+")"+'.pth')

                    global_agent.save_models('models_param/Personal_DQN/'+method+'/'+directory+'/'+file_name+"__("+str(cnt)+")"+'.pth')

                    max_reward = reward
                    cnt+=1
            print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()


np.save('npfiles/Personal_DQN/mean_reward/'+file_name+'__mean_reward.npy',np.array(mean_reward))
np.save('npfiles/Personal_DQN/reward/'+file_name+'__reward.npy',np.array(reward_hist))


plt.plot(reward_hist)
plt.savefig('figs/Personal_DQN/reward/'+file_name+'__reward.png')
#plt.show()
plt.close()

plt.plot(mean_reward)
plt.savefig('figs/Personal_DQN/mean_reward/'+file_name+'__mean_reward.png')
#plt.show()
plt.close()


