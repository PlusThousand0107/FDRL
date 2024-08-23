#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt




#PPO

dist_poly=np.load('Cent_PPO___Epochs_5_Eps_0.05_Batch_512_(1)__mean_reward.npy')
central_poly=np.load('Dist_PPO___Epochs_5_Eps_0.05_Batch_32_(4)__mean_reward.npy')
FL_poly_agg1=np.load('FL_PPO___Epochs_5_Eps_0.05_AggPer_1_Batch_32_(4)__mean_reward.npy')
FL_poly_agg10=np.load('FL_PPO___Epochs_5_Eps_0.05_AggPer_10_Batch_32_(4)__mean_reward.npy')
FL_poly_agg100=np.load('FL_PPO___Epochs_5_Eps_0.05_AggPer_100_Batch_32_(4)__mean_reward.npy')
FL_poly_agg1000=np.load('FL_PPO___Epochs_5_Eps_0.05_AggPer_1000_Batch_32_(4)__mean_reward.npy')


plt.figure(figsize=(12, 8))
plt.plot(dist_poly,linewidth=5,label="Centralized",markeredgewidth=5,markerfacecolor='w')
plt.plot(central_poly,linewidth=5,label="Distributed",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg1,linewidth=5,label="FL Aggregaion period = 1",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg10,linewidth=5,label="FL Aggregaion period = 10",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg100,linewidth=5,label="FL Aggregaion period = 100",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg1000,linewidth=5,label="FL Aggregaion period = 1000",markeredgewidth=5,mfc='none')


plt.xlabel('Iteration',fontsize=22)
plt.ylabel('Network Sum Rate',fontsize=22)
plt.grid(b=None, which='major', axis='both')
plt.title('PPO', fontsize=22)
plt.legend(loc="lower right",prop={'size': 16})
plt.xticks(size = 20)
plt.yticks(size = 20)

plt.show()