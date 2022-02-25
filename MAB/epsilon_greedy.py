#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 16:24:38 2021

@author: shishuqing
"""

import random

def ind_max(x):
  m = max(x)
  return x.index(m)

class EpsilonGreedy(object):
  def __init__(self, **config):
    self.epsilon = config['epsilon']
    self.policy_arms_number = config['policy_arms_number']
    self.total_arms_number = config['total_arms_number']
    return
    

  def initialize(self):
    '''
      Initialize count and expected value for each arm
      
      '''
    self.counts = [0 for col in range(self.total_arms_number)]
    self.values = [0.0 for col in range(self.total_arms_number)]
    return

  def select_arm(self,r=0,s=0):
    '''
      Using the epsilon-greedy algorithm to choose an action

      '''
    if random.random() > self.epsilon:
      return ind_max(self.values)
    else:
      return random.randrange(len(self.values))

  def update(self, chosen_arm, reward):
    '''
      Updating the values after spoting the enviornment;
      consider there will be multiple arms be pulled at the same time
      '''
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
    return
if __name__ == "__main__":
    
    arms = [n for n in range(100)]
    config = {
        "epsilon": 0.1,
        "policy_arms_number": 1,
        "total_arms_number": 100,
        }
    print(config['epsilon'])
    agent = EpsilonGreedy(**config)
    agent.initialize()
    print(agent.values)
    for _ in range(1000):
        n = agent.select_arm()
        agent.update(n,arms[n])
    n = agent.select_arm()
    print(n)
        








