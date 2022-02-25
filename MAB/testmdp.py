#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:08:52 2021

@author: shishuqing
"""

import mdptoolbox, mdptoolbox.example
import numpy as np
P, R = mdptoolbox.example.forest()
print(np.shape(P),np.shape(R))
fh = mdptoolbox.mdp.FiniteHorizon(P, R, 0.9, 100)
fh.run()
print(np.shape(fh.V))
