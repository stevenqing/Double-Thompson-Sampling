#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:00:07 2021

@author: shishuqing
"""

import random
import math

class Boltzmann(object):
  def __init__(self, **config):
    self.tau = config['tau']
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
    
  def random_pick(self,l,prob):
      x = random.uniform(0,1)
      cumulative_probability=0.0
      for item,item_probability in zip(l,prob):
          cumulative_probability+=item_probability
          if x < cumulative_probability:
              break
      return item
          
  def select_arm(self,r=0,s=0):
    '''
      Using the epsilon-greedy algorithm to choose an action

      '''
    prob_list,sum_prob = [],0
    
    for i in range(len(self.values)):
        prob = math.exp(self.values[i]/self.tau)
        prob_list.append(prob)
        sum_prob += prob
        
    final_prob = [prob/sum_prob for prob in prob_list]
    final_arm = [i for i in range(self.total_arms_number)]
    p = self.random_pick(final_arm,final_prob)
    
    return p
    
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
    
    arms = [1,2,3,4,5]
    config = {
        "tau": 20,
        "policy_arms_number": 1,
        "total_arms_number": 5,
        }
    agent = Boltzmann(**config)
    agent.initialize()
    print(agent.values)
    for _ in range(100):
        n = agent.select_arm()
        agent.update(n,arms[n])
    print(agent.values)
        








