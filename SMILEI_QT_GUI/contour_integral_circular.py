# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:37:11 2024

@author: jerem
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.array([-1.,-1.])
v = np.array([0.,0.])

scat = plt.scatter(x[0],x[1],s=2)
line, = plt.plot(x[0],x[1])
plt.grid()

dt = 0.01
pos_hist_x = [x[0]]
pos_hist_y = [x[1]]

L = 5
plt.xlim(-L,L)
plt.ylim(-L,L)

def temp_env(t):
    Tp = 20
    if t<Tp/2:
        return np.sin(np.pi*t/Tp)**2
    else:
        return t*0+1
    
def spatial(x):
    r = np.sqrt(x[0]**2+x[1]**2)
    return np.exp(-r**2/3)
    
t_range = np.arange(0,100,dt)
for t in t_range:
    
    acc = np.array([np.cos(t),np.sin(t)])*temp_env(t)*spatial(x)
    
    v += acc*dt
    x += v*dt
    pos_hist_x.append(x[0])
    pos_hist_y.append(x[1])

    scat.set_offsets(x)
    line.set_data(pos_hist_x,pos_hist_y)
    plt.pause(0.001)
