import numpy as np
from Environment_CU import Env_cellular
import time
import matplotlib.pyplot as plt
from PPO_agent import PPOAgent
import torch

import argparse # add parse

import os # create directory

# def parse_args():
#     parse=argparse.ArgumentParser()
#     parse.add_argument('Epochs',type=int) #
#     parse.add_argument('Eps',type=float) #
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
    power_num = 10  #action_num
    seed=11
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
    
    max_reward = 0
    batch_size = 512
    max_episode = 7000

    Ns = 11
    env.set_Ns(Ns) 

    load_checkpoint = False
    state_num = env.state_num # 50
    action_num = env.power_num # 10

    # args=parse_args()
    # Epochs=args.Epochs
    # Eps=args.Eps
    # Num=args.Num
    Epochs=5
    Eps=0.05
    Num=1
    
    agent = PPOAgent(epochs=Epochs,eps=Eps,state_dims=state_num,gamma=0.9,n_actions=action_num) # alr=0.0003, clr=0.001

    interval = 100
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]

    file_name="Cent_PPO___Epochs_"+str(Epochs)+"_Eps_"+str(Eps)+"_Batch_"+str(batch_size)+"_("+str(Num)+")"

    path = os.path.join('models_param/PPO/', file_name)
    os.mkdir(path)

    cnt=0
    max_cnt=0
    
    for k in range(max_episode):

        agent.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []} #

        reward_policy_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            a = agent.choose_action(s_actor)
            
            p = env.get_power_set(min_p)[a]

            s_actor_next, _, rate, r ,_= env.step(p)

            for j in range(100):
                agent.transition_dict['states'].append(s_actor[j])
                agent.transition_dict['actions'].append(a[j])
                agent.transition_dict['next_states'].append(s_actor_next[j])
                agent.transition_dict['rewards'].append(rate[j])

            agent.store_rewards(rate)
            
            s_actor = s_actor_next
            reward_policy_list.append(r)
            all_reward.append(r)
            
        #agent.learn()
        agent.update(agent.transition_dict,batch_size)
            
        reward_hist.append(np.mean(reward_policy_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)

            if reward > 1.58 and k>4000:
                agent.save_models("models_param/PPO/"+file_name+"/"+file_name)
                cnt+=1
                if reward > max_reward:
                   max_reward = reward
                   max_cnt=cnt

            print("Episode(train):%d  policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
            st = time.time()

print(file_name+" : "+str(max_reward)+" max cnt : "+str(max_cnt))
np.save('npfiles/PPO/mean_reward/'+file_name+'__mean_reward.npy',np.array(mean_reward))
np.save('npfiles/PPO/reward/'+file_name+'__reward.npy',np.array(reward_hist))


plt.plot(reward_hist)
plt.savefig('figs/PPO/reward/'+file_name+'__reward.png')
#plt.show()
plt.close()

plt.plot(mean_reward)
plt.savefig('figs/PPO/mean_reward/'+file_name+'__mean_reward.png')
#plt.show()
plt.close()


