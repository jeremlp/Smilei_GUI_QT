# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:07:49 2025

@author: Jeremy
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

dt = 0.1
Size = 10
Window = 200


def update(POS, OLD_POS):
    POS,OLD_POS = updatePositions(POS, OLD_POS)
    POS,OLD_POS = applyConstraint(POS, OLD_POS)
    POS = applyCollision(POS)
    return POS,OLD_POS

def updatePositions(POS,OLD_POS):
    
    for i in range(len(POS)):
        pos, old_pos = POS[i], OLD_POS[i]
        POS[i], OLD_POS[i] = updateSinglePos(pos, old_pos)
    
    return POS, OLD_POS
        
    
def updateSinglePos(pos, old_pos):
    vel = pos - old_pos
    old_pos = pos
    
    total_acc = getAcc(pos)

    pos += vel + total_acc*dt**2
    return pos, old_pos

def getAcc(pos):
    direction = pos - mouse_pos
    distance = np.linalg.norm(direction) + 0.1  # Avoid division by zero
    force = 8000 / (distance)  # Inverse square law
    mouse_acc = np.array([0,0])
    if distance < Window / 3:
            mouse_acc = (direction / distance) * force  # Normalize and apply force
    
    print(mouse_acc)
    return gravity_acc + mouse_acc
    

def applyConstraint(POS, OLD_POS):
    
    for i in range(len(POS)):
        new_pos = POS[i]
        
        if new_pos[1] - Size < 0: #Bounce on ground
            new_pos = np.array([new_pos[0],Size])
        
        if new_pos[0] - Size < 0: #left wall
            new_pos = np.array([Size,new_pos[1]])
        if new_pos[0] + Size > Window: #right wall
            new_pos = np.array([Window-Size,new_pos[1]])
        POS[i] = new_pos
    
    
    return POS, OLD_POS

def applyCollision(POS):
    
    for i in range(len(POS)):
        pos1 = POS[i]
        for j in range(len(POS)):
            if i==j: continue
            pos2 = POS[j]
            axis = pos1-pos2
            dist = np.sqrt(axis[0]**2+axis[1]**2)
            if ((dist < 2*Size) & (dist>0.001)):
                u = axis/dist
                delta = 2*Size - dist
                coef = 0.5 #viscosity
                POS[i] += 0.5*delta*u*coef
                POS[j] -= 0.5*delta*u*coef
    return POS
            


fig, ax = plt.subplots()
N = 50
POS = np.random.random((N,2))*Window/2
OLD_POS = POS

circles = []
for _ in range(N):
    circle = patches.Circle((np.random.uniform(1, Window), np.random.uniform(1, Window)), Size, fc='C0',ec="k")
    ax.add_patch(circle)
    circles.append(circle)
plt.grid()
plt.xlim(0,Window)
plt.ylim(0,Window)
plt.pause(0.9)

gravity_acc = np.array([0,-50])

mouse_pos = np.array([Window / 2, Window / 2])

def on_mouse_move(event):
    global mouse_pos
    if event.xdata is not None and event.ydata is not None:
        mouse_pos = np.array([event.xdata, event.ydata])
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)



for t in range(10000):
    print(t)
    POS, OLD_POS = update(POS, OLD_POS)
    
    for i, circle in enumerate(circles):
        circle.center = POS[i]
    
    plt.pause(0.01)
    
    
