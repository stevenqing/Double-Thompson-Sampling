#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 17:12:49 2021

@author: shishuqing
"""

import random
import numpy as np
class ThompsonSampling(object):
    def __init__(self,**config):
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
        return

    def random_pick(self,l,prob):
        x = random.uniform(0,1)
        cumulative_probability=0.0
        for item,item_probability in zip(l,prob):
            cumulative_probability+=item_probability
            if x < cumulative_probability:
                break
        return item
    
    
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
        if self.policy_arms_number == 1:
            self.alpha[chosen_arm] += reward
            self.beta[chosen_arm] = self.counts[chosen_arm]+reward
        else:
            for i in range(self.policy_arms_number):
                self.values[chosen_arm[i]][0] += reward[i]
                self.values[chosen_arm[i]][1] = self.counts[chosen_arm[i]] - reward[i]
        return 
        
if __name__ == "__main__":
    
    arms = [n for n in range(5)]
    config = {
        "policy_arms_number": 1,
        "total_arms_number": 5,
        }
    agent = ThompsonSampling(**config)
    agent.initialize()
    for _ in range(5000):
        n = agent.select_arm()
        agent.update(n,arms[n])
    print(agent.alpha,agent.beta)
    n = agent.select_arm()
    print(n)
    print(np.array(agent.alpha)/(np.array(agent.alpha)+np.array(agent.beta)))
        
        
        
        