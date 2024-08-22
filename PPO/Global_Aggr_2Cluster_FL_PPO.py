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
#     parse.add_argument('AggPer',type=int) #
#     parse.add_argument('Num',type=int)
#     args=parse.parse_args()
#     return args

if __name__ == '__main__':
    fd = 10 
    Ts = 20e-3 
    n_x = 5 #5
    n_y = 5 #5
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
    batch_size = 32 #32
    max_episode = 7000 

    Ns = 11
    env.set_Ns(Ns) 

    load_checkpoint = False
    state_num = env.state_num
    action_num = env.power_num 
    
    num_of_agents=n_x*n_y
    Agents_list=[]
    reward_lists_of_list=[]
    mean_reward_lists_of_list=[]

    # args=parse_args()
    # Epochs=args.Epochs
    # Eps=args.Eps
    # AggPer=args.AggPer
    # Num=args.Num
    Epochs=5
    Eps=0.05
    AggPer=10
    Num=1
    
    Global_agents_list=[]
    global_agent_G = PPOAgent(epochs=Epochs,eps=Eps,state_dims=state_num,gamma=0.9,n_actions=action_num)
    global_agent_1 = PPOAgent(epochs=Epochs,eps=Eps,state_dims=state_num,gamma=0.9,n_actions=action_num)
    global_agent_2 = PPOAgent(epochs=Epochs,eps=Eps,state_dims=state_num,gamma=0.9,n_actions=action_num)

    Global_agents_list.append(global_agent_1)
    Global_agents_list.append(global_agent_2)


    for n in range(num_of_agents):
        Agents_list.append(PPOAgent(epochs=Epochs,eps=Eps,state_dims=state_num,gamma=0.9,n_actions=action_num))
        reward_lists_of_list.append([])
        mean_reward_lists_of_list.append([])


    interval = 100

    
    st = time.time()
    reward_hist = list()
    all_reward=[]
    mean_reward=[]

    a=np.zeros(maxM*num_of_agents).astype(np.int64) #
    p=np.zeros(maxM*num_of_agents) #
    agent_rewards=np.zeros((num_of_agents,max_episode)) #
   
    file_name="Global_Aggr_2Cluster_FL_PPO___Epochs_"+str(Epochs)+"_Eps_"+str(Eps)+"_AggPer_"+str(AggPer)+"_Batch_"+str(batch_size)+"_("+str(Num)+")"

    path = os.path.join('models_param/PPO/', file_name)
    os.mkdir(path)

    cnt=0
    max_cnt=0

    n1=[0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24] # 16
    n2=[6, 7, 8, 11, 12, 13, 16, 17, 18] # 9

    for k in range(max_episode):

        for n in range(num_of_agents): #
            agent=Agents_list[n]
            agent.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []} #

        reward_dqn_list = []
        s_actor, _ = env.reset()
        for i in range(int(Ns)-1):
            for n in range(num_of_agents):
                s_actor_agent=s_actor[n*maxM:(n+1)*maxM] #
                agent=Agents_list[n]
                a_agent = agent.choose_action(s_actor_agent)
                
                p_agent=env.get_power_set(min_p)[a_agent]
                
                a[n*maxM:(n+1)*maxM]=a_agent #
                p[n*maxM:(n+1)*maxM]=p_agent #
                for j in range(4):
                    agent.transition_dict['states'].append(s_actor_agent[j])
                    agent.transition_dict['actions'].append(a_agent[j])

                
            s_actor_next, _, rate, r,rate_all_agents = env.step(p)
            
            for n in range(num_of_agents): #
                agent=Agents_list[n]
                rate_agent=rate[n*maxM:(n+1)*maxM]
                s_actor_next_agent=s_actor_next[n*maxM:(n+1)*maxM]
                agent.store_rewards(rate_agent)
                reward_lists_of_list[n].append(np.mean(rate_all_agents[n*maxM:(n+1)*maxM]))
                for j in range(4):
                    agent.transition_dict['next_states'].append(s_actor_next_agent[j])
                    agent.transition_dict['rewards'].append(rate_agent[j])
                
            s_actor = s_actor_next
            reward_dqn_list.append(r)
            all_reward.append(r)
            
        for n in range(num_of_agents): #
            agent=Agents_list[n]
            #agent.learn()
            agent.update(agent.transition_dict,batch_size)
            agent_rewards[n,k] =np.mean(reward_lists_of_list[n][-(Ns-1):])
            
            
        if k % AggPer == 0:
            # 1
            global_dict_actor_1 = global_agent_1.actor.state_dict()
            global_dict_critic_1 = global_agent_1.critic.state_dict()

            for kd in global_dict_actor_1.keys():
                global_dict_actor_1[kd] = torch.stack([Agents_list[n].actor.state_dict()[kd] for n in n1 ], 0).mean(0)
            for kd in global_dict_critic_1.keys():
                global_dict_critic_1[kd] = torch.stack([Agents_list[n].critic.state_dict()[kd] for n in n1 ], 0).mean(0)

            global_agent_1.actor.load_state_dict(global_dict_actor_1)
            global_agent_1.critic.load_state_dict(global_dict_critic_1)


            # 2
            global_dict_actor_2 = global_agent_2.actor.state_dict()
            global_dict_critic_2 = global_agent_2.critic.state_dict()

            for kd in global_dict_actor_2.keys():
                global_dict_actor_2[kd] = torch.stack([Agents_list[n].actor.state_dict()[kd] for n in n2 ], 0).mean(0)
            for kd in global_dict_critic_2.keys():
                global_dict_critic_2[kd] = torch.stack([Agents_list[n].critic.state_dict()[kd] for n in n2 ], 0).mean(0)


            global_agent_2.actor.load_state_dict(global_dict_actor_2)
            global_agent_2.critic.load_state_dict(global_dict_critic_2)


            # G
            global_dict_actor_G = global_agent_G.actor.state_dict()
            global_dict_critic_G = global_agent_G.critic.state_dict()

            for kd in global_dict_actor_G.keys():
                global_dict_actor_G[kd] = torch.stack([Global_agents_list[n].actor.state_dict()[kd] for n in range(2) ], 0).mean(0)
            for kd in global_dict_critic_G.keys():
                global_dict_critic_G[kd] = torch.stack([Global_agents_list[n].critic.state_dict()[kd] for n in range(2) ], 0).mean(0)

            global_agent_G.actor.load_state_dict(global_dict_actor_G)
            global_agent_G.critic.load_state_dict(global_dict_critic_G)

            for n in range(num_of_agents):
                Agents_list[n].actor.load_state_dict(global_agent_G.actor.state_dict())
                Agents_list[n].critic.load_state_dict(global_agent_G.critic.state_dict())

            
        reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
        if k % interval == 0: 
            reward = np.mean(reward_hist[-interval:])
            mean_reward.append(reward)
            for n in range(num_of_agents): #
                mean_reward_lists_of_list[n].append(np.mean(agent_rewards[n,-interval:]))

            if reward > 1.55 and k>5700:
                global_agent_G.save_models("models_param/PPO/"+file_name+"/"+file_name+"_global_agentG__("+str(cnt)+")")
                cnt+=1
                if reward > max_reward:
                   max_reward = reward
                   max_cnt=cnt

            print("Episode(train):%d  Multi_agent Policy: %.3f  Time cost: %.2fs" %(k, reward, time.time()-st))
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


