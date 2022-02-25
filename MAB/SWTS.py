#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:12:57 2021

@author: shishuqing
"""


import numpy as np
class SWTS(object):
    def __init__(self,**config):
        self.sliding_window_size = config['sliding_window_size']
        self.policy_arms_number = config['policy_arms_number']
        self.total_arms_number = config['total_arms_number']
        return
    
    def initialize(self):
        '''
        Initialize count and expected value for each arm
      
        '''
        self.counts = [0 for col in range(self.total_arms_number)]
        self.alpha = [1 for col in range(self.total_arms_number)]
        self.beta = [1 for col in range(self.total_arms_number)]
        self.last_n_actions = []
        return
    
    
    def select_arm(self,r=0,s=0):
        if self.policy_arms_number == 1:
            samples = [np.random.beta(self.alpha[x], self.beta[x]) for x in range(self.total_arms_number)]
            i = max(range(self.total_arms_number), key=lambda x: samples[x])
            self.counts[i]+=1
            return i
        else:
            final_value = []
            for i in range(self.total_arms_number):
                expected_value = self.values[i][0]/(self.values[i][0]+self.values[i][1])
                final_value.append(expected_value)
            final_value.sort().reverse()
            return final_value[:self.policy_arms_number]
        
    
    def update(self,chosen_arm,reward):
        self.update_last_n_actions(chosen_arm,reward)
        if self.policy_arms_number == 1:
            self.alpha[chosen_arm] += reward
            self.beta[chosen_arm] = self.counts[chosen_arm]+reward
        else:
            for i in range(self.policy_arms_number):
                self.values[chosen_arm[i]][0] += reward[i]
                self.values[chosen_arm[i]][1] = self.counts[chosen_arm[i]] + reward[i]
        return 
        
    def update_last_n_actions(self,chosen_arm,reward):
        if len(self.last_n_actions) >= self.sliding_window_size:
            tmp_action, tmp_reward = self.last_n_actions.pop(0)
            self.alpha[tmp_action] -= tmp_reward  # Update alpha
        self.last_n_actions.append((chosen_arm, reward))
if __name__ == "__main__":
    
    arms = [1,2,3,4,5]
    config = {
        "sliding_window_size":2,
        "policy_arms_number": 1,
        "total_arms_number": 5,
        }
    agent = SWTS(**config)
    agent.initialize()
    for _ in range(1000):
        n = agent.select_arm()
        agent.update(n,arms[n])
    print(agent.alpha,agent.beta)
    n = agent.select_arm()
    print(n)
        
        
        
        