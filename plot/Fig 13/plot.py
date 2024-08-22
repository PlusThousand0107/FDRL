#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt



# 1000

dist_poly=np.load('FL_DQN_Fed_AggPer_1000_(2)__mean_reward.npy')
central_poly=np.load('FL_DPG_Fed_AggPer_1000_(1)__mean_reward.npy')
FL_poly_agg1=np.load('FL_PPO___Epochs_5_Eps_0.05_AggPer_1000_Batch_32_(4)__mean_reward.npy')
FL_poly_agg10=np.load('FL_PPO___Epochs_5_Eps_0.1_AggPer_1000_Batch_32_(4)__mean_reward.npy')
FL_poly_agg100=np.load('FL_PPO___Epochs_5_Eps_0.15_AggPer_1000_Batch_32_(5)__mean_reward.npy')
FL_poly_agg1000=np.load('FL_PPO___Epochs_5_Eps_0.2_AggPer_1000_Batch_32_(4)__mean_reward.npy')


plt.figure(figsize=(12, 8))
plt.plot(dist_poly,linewidth=5,label="DQN",markeredgewidth=5,markerfacecolor='w')
plt.plot(central_poly,linewidth=5,label="DPG",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg1,linewidth=5,label="PPO (eps = 0.05)",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg10,linewidth=5,label="PPO (eps = 0.1)",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg100,linewidth=5,label="PPO (eps = 0.15)",markeredgewidth=5,markerfacecolor='w')
plt.plot(FL_poly_agg1000,linewidth=5,label="PPO (eps = 0.2)",markeredgewidth=5,mfc='none')


plt.xlabel('Iteration',fontsize=22)
plt.ylabel('Network Sum Rate',fontsize=22)
plt.grid(b=None, which='major', axis='both')
plt.title('Aggregation Period = 1000', fontsize=22)
plt.legend(loc="lower right",prop={'size': 16})
plt.xticks(size = 20)
plt.yticks(size = 20)


plt.show()
