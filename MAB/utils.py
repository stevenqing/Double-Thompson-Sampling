#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 20:59:56 2021

@author: shishuqing
"""
import math

def MSE(a,b):
    '''
    a,b为list
    len(a) = len(b)
    '''
    total = 0
    for i in range(len(a)):
        s = (a[i] - b[i])**2
        total += s
    return math.sqrt(total)

