import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

seed=11
T.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dims, 128) #128
        self.fc2 = nn.Linear(128, 64) # 128 64
        self.fc3 = nn.Linear(64, n_actions) #64
        #self.actor_optimizer = optim.Adam(self.parameters(), lr=alr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = self.fc3(x)

        return F.softmax(self.fc3(x), dim=1, dtype=T.double)
###
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128) #128
        self.fc2 = nn.Linear(128, 64) # 128 64
        self.fc3 = nn.Linear(64, 1) #64
        #self.critic_optimizer = optim.Adam(self.parameters(), lr=clr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
###

class PPOAgent():
    def __init__(self, epochs, eps, state_dims, gamma, n_actions):
        
        self.gamma = gamma
        self.alr = 0.0003 # 0.0003
        self.clr = 0.001 # 0.001
        self.lmbda = 0.97 # 0.95(1.550) 0.97(1.596) 0.99(1.582) 0.98(1.259) 0.96(1.300) 0.9(1.575)
        self.epochs = epochs # 一条序列的数据用来训练轮数 10 5
        self.eps = eps  # PPO中截断范围的参数 0.2 0.15
        self.device = T.device("cuda:0") if T.cuda.is_available() else T.device("cpu")
    
        self.reward_memory = []
        self.action_memory = []

        # loss
        self.actor_loss_hist = []
        self.critic_loss_hist = []


        self.actor = PolicyNetwork(state_dims, n_actions)
        self.critic = ValueNet(state_dims)

        self.actor_optimizer = T.optim.Adam(self.actor.parameters(),
                                                lr=self.alr)
        self.critic_optimizer = T.optim.Adam(self.critic.parameters(),
                                                 lr=self.clr)
        
        self.transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []} #



    
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.device).squeeze()
        probabilities = self.actor(state)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        #log_probs = action_probs.log_prob(action)
        # self.action_memory.append(log_probs)

        return action.detach().cpu().numpy()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
        

    def normalize_rewards(reward_list): # ??
        
        reward_arr=np.asarray(reward_list)
        mean_reward=np.mean(reward_arr,axis=0)
        std_reward=np.std(reward_arr,axis=0)
        normalized_reward=(reward_arr-mean_reward)/(std_reward+1e-8)
        
        return normalized_reward
   

    
    def save_models(self,save_file):
        print('...save model....')
        T.save(self.actor, save_file+"_actor.pth")
        T.save(self.critic, save_file+"_critic.pth")


    def load_model(self,path): #
        print('...load model....')
        self.actor=T.load(path)

    

    def compute_advantage(self,gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return T.tensor(advantage_list, dtype=T.float)

    
    def update(self, transition_dict, batch_size):



        sum_actor_loss = 0.0
        sum_critic_loss = 0.0

        states = T.tensor(transition_dict['states'],dtype=T.float).to(self.device)
        actions = T.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = T.tensor(transition_dict['rewards'],dtype=T.float).view(-1, 1).to(self.device)
        next_states = T.tensor(transition_dict['next_states'],dtype=T.float).to(self.device)


        # NORM
        rewards=PPOAgent.normalize_rewards(transition_dict['rewards'])
        rewards = T.tensor(rewards,dtype=T.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)

        advantage = self.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)

        old_log_probs = T.log(self.actor(states).gather(1,actions)).detach()
        


        for _ in range(self.epochs):
            if batch_size>40:
                numbers=list(range(1000))
            else : numbers=list(range(40))

            selected_numbers=random.sample(numbers,batch_size) #
            #print(selected_numbers)

            batch_states=[states[i] for i in selected_numbers]
            tensor_batch_states=T.stack(batch_states).to(self.device)
            
            batch_actions=[actions[i] for i in selected_numbers]
            tensor_batch_actions=T.stack(batch_actions).to(self.device)

            batch_old_log_probs=[old_log_probs[i] for i in selected_numbers]
            tensor_batch_old_log_probs=T.stack(batch_old_log_probs).to(self.device)

            batch_advantage=[advantage[i] for i in selected_numbers]
            tensor_batch_advantage=T.stack(batch_advantage).to(self.device)

            batch_td_target=[td_target[i] for i in selected_numbers]
            tensor_td_target=T.stack(batch_td_target).to(self.device)



            tensor_batch_log_probs = T.log(self.actor(tensor_batch_states).gather(1, tensor_batch_actions))
            ratio = T.exp(tensor_batch_log_probs - tensor_batch_old_log_probs)
            surr1 = ratio * tensor_batch_advantage
            surr2 = T.clamp(ratio, 1 - self.eps,1 + self.eps) * tensor_batch_advantage  # 截断
            actor_loss = T.mean(-T.min(surr1, surr2))  # PPO损失函数
            critic_loss = T.mean(F.mse_loss(self.critic(tensor_batch_states), tensor_td_target.detach()))

            sum_actor_loss += actor_loss
            sum_critic_loss += critic_loss
            # self.all_actor_loss_hist.append(actor_loss.data.cpu()) # to cpu
            # self.all_critic_loss_hist.append(critic_loss.data.cpu()) # to cpu

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_loss_hist.append(sum_actor_loss.data.cpu()) # to cpu
        self.critic_loss_hist.append(sum_critic_loss.data.cpu()) # to cpu

