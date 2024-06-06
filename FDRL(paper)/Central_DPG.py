#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 18:58:12 2020

@author: peymantehrani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:13:15 2020

@author: peymantehrani
"""

import numpy as np
from DQN_agent_pytorch import DQNAgent
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
from Reinforce_Pytorch import PolicyGradientAgent
import torch
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
    power_num = 10  #action_num

    
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    batch_size = 500
    max_episode = 7000
    buffer_size = 50000
    Ns = 11
    env.set_Ns(Ns) 

    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num
        
    agent = PolicyGradientAgent(lr=0.0003,state_dims=state_num,gamma=0.99,n_actions=action_num)

    interval = 100
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]

    file_name="Central_DPG"
    
    for k in range(max_episode):
        reward_policy_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            a = agent.choose_action(s_actor)

            p=env.get_power_set(min_p)[a]

            s_actor_next, _, rate, r ,_= env.step(p)

            agent.store_rewards(rate)

            s_actor = s_actor_next
            reward_policy_list.append(r)
            all_reward.append(r)
            

        agent.learn()
            
        reward_hist.append(np.mean(reward_policy_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            if reward > max_reward:
                agent.save_model('models_param/DPG/'+file_name+'.pth')
                max_reward = reward
            print("Episode(train):%d  policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()


np.save('npfiles/DPG/mean_reward/'+file_name+'__mean_reward.npy',np.array(mean_reward))
np.save('npfiles/DPG/reward/'+file_name+'__reward.npy',np.array(reward_hist))
loss_hist=agent.get_loss()
np.save('npfiles/DPG/loss/'+file_name+'__loss.npy',np.array(loss_hist))


# mean loss
loss_mean_hist=0
mean_loss=[]
for i in range(len(loss_hist)):
    loss_mean_hist+=loss_hist[i]
    if i%100==0:
        mean_loss.append(loss_mean_hist/100)
        loss_mean_hist=0

np.save('npfiles/DPG/mean_loss/'+file_name+'__mean_loss.npy',np.array(mean_loss))



### plt
plt.plot(loss_hist)
plt.savefig('figs/DPG/loss/'+file_name+'__loss.png')
#plt.show()
plt.close()

plt.plot(mean_loss)
plt.savefig('figs/DPG/mean_loss/'+file_name+'__mean_loss.png')
#plt.show() 
plt.close() 


plt.plot(reward_hist)
plt.savefig('figs/DPG/reward/'+file_name+'__reward.png')
#plt.show()
plt.close()

plt.plot(mean_reward)
plt.savefig('figs/DPG/mean_reward/'+file_name+'__mean_reward.png')
#plt.show()
plt.close()


