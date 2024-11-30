# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:04:16 2024

@author: jerem
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate
plt.close("all")
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A0.3_HD/opt_a0.3_dx48')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","By","By","Bz"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1


track_N_tot = T0.nParticles
t_range = T0.getTimes()

track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1



x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py

tmax = 120
Nid = 0
def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def f_prime(r,z):
    C_lp = np.sqrt(1/math.factorial(abs(l1)))
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l1)-1) * (-2*r**2+w(z)**2*abs(l1))
def f_squared_prime(r,z):
    return 2*w0**2/(w(z)**2*r) * f(r,z)**2*(abs(l1)-2*(r/w0)**2+ 4*(z**2/w0**4))

def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr
@njit
def Ftheta(r,theta,z):
    #-1 bcs Ftheta uses IntBz instead of IntEz, so need to inverse sign
    expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
          a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) + 
                          4 * (2 * r**2 * z * cos(theta)**2 + 
                               2 * r**4 * cos(theta)**3 * sin(theta) - 
                               2 * r**2 * z * sin(theta)**2 + 
                               2 * r**4 * cos(theta) * sin(theta)**3))) / 
         (2 * r * (w0**4 + 4 * z**2)**3))
    return expr
@njit
def FthetaCart(x,y,z):
    #/!\ USES Z AS PROPAGATION DIRECTION /!\

    #-1 bcs Ftheta uses IntBz instead of IntEz, so need to inverse sign

    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
          a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) + 
                          4 * (2 * r**2 * z * cos(theta)**2 + 
                               2 * r**4 * cos(theta)**3 * sin(theta) - 
                               2 * r**2 * z * sin(theta)**2 + 
                               2 * r**4 * cos(theta) * sin(theta)**3))) / 
         (2 * r * (w0**4 + 4 * z**2)**3))
    return expr

Nid = np.where(np.abs(Lx_track[-1])==np.max(np.abs(Lx_track[-1])))[0][0]
gamma = sqrt(1+a0**2)
x_pos = 5*l0

x0 = np.mean(x[0])
x_range = np.arange(-1*w0,1*w0,0.1)
X,Y = np.meshgrid(x_range,x_range)
extent = [x_range[0]/l0,x_range[-1]/l0,x_range[0]/l0,x_range[-1]/l0]

# plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=1)

r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R,THETA = np.meshgrid(r_range,theta_range)
Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
# plt.plot(r_range/l0,Lx_max_model,"k--",alpha=1)
# plt.plot(r_range/l0,-Lx_max_model,"k--",alpha=1, label="Model $L_z^{(2)}$")


# plt.axvline(w0/l0,color="k",ls="--",label="w0")
# plt.axvline(w0/sqrt(2)/l0,color="red",ls="--",label="w0/sqrt(2)")
# plt.grid()
# plt.legend()


plt.figure()
plt.imshow(FthetaCart(X,Y,x0),extent=extent,cmap="RdBu")
plt.colorbar()
plt.plot(y[:,Nid]/l0,z[:,Nid]/l0,"k")
t_max_idx_interaction = np.where(t_range<(x_pos+Tp))[0][-1]
plt.plot(y[:t_max_idx_interaction,Nid]/l0,z[:t_max_idx_interaction,Nid]/l0,"r--")

plt.scatter(y[0,Nid]/l0,z[0,Nid]/l0,marker="o",color="k")

max_intensity = plt.Circle((0,0),w0/sqrt(2)/l0,fill=False,ec="red")
plt.gca().add_patch(max_intensity)
plt.scatter(0,0,marker="x",color="red")
plt.grid()
plt.tight_layout()

# plt.figure()
ftheta_time =FthetaCart(y[:,Nid],z[:,Nid],x[:,Nid])*(sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0))**2
# plt.plot(t_range/l0,ftheta_time,".-")
# plt.grid()
# plt.xlabel("t/t0")

Lx_max_model = np.max(LxEpolar(r[0,Nid],theta[0,Nid],z[0,Nid],w0,a0,3/8*Tp))
Lx_integrated = integrate.simpson(r[:,Nid]*ftheta_time,x=t_range)
print("Model:",Lx_max_model)
print("Model integrated:",Lx_integrated)
print("Sim:",Lx_track[-1,Nid])




def dr_long_time(r,t):
    pr_end = -a0**2/4*f_squared_prime(r, x_pos)*3/8*Tp
    return pr_end*(t-Tp-x_pos)
    
# def dr_short_time(r,t):
#     fr = -a0**2/4*f_squared_prime(r, x_pos)*(sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x_pos)<Tp)*((t_range-x_pos)>0))**2
#     pr = -a0**2/4*f_squared_prime(r, x_pos) * integral(integral(sin2))
    
#     return dr

def dr(r,t):
    dr = 1* -(3/8)*a0**2/4*f_squared_prime(r, x_pos)*t**2/2 * (t>2)
    return dr

# integrate.simpson(dr)

# plt.close("all")
plt.figure()
plt.plot(t_range/l0,r[:,Nid],".-")
# plt.plot(t_range/l0, r[0,Nid]+dr_short_time(r[0,Nid], t_range))
plt.plot(t_range/l0, r[0,Nid]+dr(r[0,Nid],t_range-x_pos))
plt.grid()

r0 = r[0,Nid]
theta0 = theta[0,Nid]
Lx_integrated_R = integrate.simpson((r0+dr(r0,t_range)) * Ftheta(r0+dr(r0,t_range), theta0, x_pos)*(sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0))**2,x=t_range)
print(Lx_integrated_R)

plt.figure()
temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
plt.plot(t_range/l0,r[:,Nid]*ftheta_time,".-",label="r*Ftheta exact")
plt.plot(t_range/l0,(r0) * Ftheta(r0, theta0, x_pos)*temp_env**2,label="r0*Ftheta0 NR")
plt.plot(t_range/l0,(r0+dr(r0,t_range-x_pos)) * Ftheta(r0+dr(r0,t_range-x_pos), theta0, x_pos)*temp_env**2,label="r*Ftheta R")
# plt.plot(t_range/l0,r[:,Nid] * Ftheta(r[:,Nid], theta0, x_pos)*temp_env,label="r*Ftheta R")

plt.grid()
plt.legend()
# @njit
def Lx_distrib(use_R=False,use_Z=False,dt=0.1):
    t_range_lx = np.arange(0,t_range[-1],dt)
    temp_env = sin(pi*(t_range_lx-x_pos)/Tp)**2*((t_range_lx-x_pos)<Tp)*((t_range_lx-x_pos)>0)

    Lx_r_range = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson((r+dr(r,t_range_lx-x_pos)*use_R) * Ftheta(r+dr(r,t_range_lx-x_pos)*use_R, theta, x_pos+0.25*t_range_lx*use_Z)*temp_env**2, x=t_range_lx)
            # LxR_distrib = integrate.simpson(r * Ftheta(r, theta, x_pos)*temp_env, x=t_range_lx)

            Lx_theta_list.append(LxR_distrib)
        Lx_r_range.append(np.max(Lx_theta_list))
    return np.array(Lx_r_range)

plt.figure()
plt.scatter(r[0]/l0,Lx_track[-1],s=1,alpha=0.25)

Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
plt.plot(r_range/l0,Lx_max_model,"k--",alpha=1)
plt.plot(r_range/l0,-Lx_max_model,"k--",alpha=1, label="Model $L_z^{(2)}$")

Lx_r_range = Lx_distrib()
plt.plot(r_range/l0,Lx_r_range,"C1")
plt.plot(r_range/l0,-Lx_r_range,"C1",label="Lx2 time int")
plt.axvline(w0/sqrt(2)/l0,color="r",ls="--")

# Lx_r_range = Lx_distrib(use_R=1)
# plt.plot(r_range/l0,Lx_r_range,"r")
# plt.plot(r_range/l0,-Lx_r_range,"r",label="Lx R time int")
# Lx_r_range = Lx_distrib(use_R=1,use_Z=1)
# plt.plot(r_range/l0,Lx_r_range,"C2")
# plt.plot(r_range/l0,-Lx_r_range,"C2",label="Lx R,Z time int")
plt.grid()
plt.legend()


# plt.figure()
# temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
# plt.plot(t_range/l0,r[:,Nid]*ftheta_time*temp_env,".-",label="r*Ftheta exact")
# plt.plot(t_range/l0,r0*Ftheta(r0, theta0, x_pos)*temp_env,label="r0*Ftheta0 NR")
# plt.plot(t_range/l0,(r0+dr(r0,t_range)) * Ftheta(r0+dr(r0,t_range), theta0, x_pos)*temp_env,label="r*Ftheta R")
# plt.legend()
# plt.grid()
# plt.figure()
# plt.scatter(r[0]/l0,pr[-1],s=1)
# pr_end = -a0**2/4*f_squared_prime(r_range, x_pos)*3/8*Tp
# plt.plot(r_range/l0, pr_end,"r--")
# # plt.plot(r_range/l0,pr_end*(t_range[-1]-Tp),"r--")            
# plt.grid()

# plt.figure()
# plt.scatter(r[0]/l0,r[-1]-r[0],s=1)
# # plt.plot(r_range/l0, pr_end,"r--")
# plt.plot(r_range/l0,pr_end*(t_range[-1]-Tp-x_pos),"r--")            
# plt.grid()