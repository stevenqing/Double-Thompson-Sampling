#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 20:40:44 2021

@author: shishuqing
"""

import math
 
 
def find_index(x):
    return x.index(max(x))
 
 
class UCB1():
    def __init__(self, **config):
        self.policy_arms_number = config['policy_arms_number']
        self.total_arms_number = config['total_arms_number']
        return
 
    def initialize(self):
        '''
        Initialize count and expected value for each arm
        '''
        self.counts = [0 for col in range(self.total_arms_number)]
        self.values = [0.0 for col in range(self.total_arms_number)]
 
    def select_arm(self,r=0,s=0):
        '''
        Using the UCB algorithm to choose an action
        '''
        
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        ucb_value = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
 
        for arm in range(n_arms):
            bonus = math.sqrt((2*math.log(total_counts))/float(self.counts[arm]))
            ucb_value[arm] = self.values[arm] + bonus
            return find_index(ucb_value)
 
    def update(self, chosen_arm, reward):
        if self.policy_arms_number == 1:
            self.counts[chosen_arm] = self.counts[chosen_arm] + 1
            n = self.counts[chosen_arm]
    
            value = self.values[chosen_arm]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[chosen_arm] = new_value
        else:
            reward_mean = reward / self.policy_arms_number
            for i in range(len(chosen_arm)):
                self.counts[i] = self.counts[i] + 1
                new_value = ((i - 1) / float(n)) * self.values[i] + (i / float(n)) * reward_mean
                self.values[i] = new_value

if __name__ == "__main__":
    arms = [1,2,3,4,5]
    config = {
        "policy_arms_number": 1,
        "total_arms_number": 5,
        }
    agent = UCB1(**config)
    agent.initialize()
    print(agent.values)
    for _ in range(100):
        n = agent.select_arm()
        agent.update(n,arms[n])
    print(agent.values)