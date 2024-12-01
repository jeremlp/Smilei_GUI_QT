# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:04:16 2024

@author: jerem
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = r'C:\Users\jerem\Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate
plt.close("all")

requested_a0 = 2
if requested_a0==2:
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/opt_a2.0_dx64')
else:
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/opt_a0.3_dx48')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","Bx","By","Bz"])

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

Ey = track_traj["Ey"][:,::N_part]
Ez = track_traj["Ez"][:,::N_part]
Ex = track_traj["Ex"][:,::N_part]
By = track_traj["By"][:,::N_part]
Bz = track_traj["Bz"][:,::N_part]
Bx = track_traj["Bx"][:,::N_part]


r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)

tmax = 120
Nid = 0
def min_max(X,Y,dr_av=0.6):
    M = []
    m = []
    da = 0.05
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    M = np.empty(a_range.shape)
    m = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        M[i] = np.nanmax(Y[mask])
        m[i] = np.nanmin(Y[mask])
    return a_range,m,M

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
# gamma = sqrt(1+a0**2)
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
plt.imshow(FthetaCart(X,Y,x0),extent=extent,cmap="RdBu",label="$F\Theta$")
plt.colorbar()
plt.plot(y[:,Nid]/l0,z[:,Nid]/l0,"k",label="Max Lx eon trajectory")
t_max_idx_interaction = np.where(t_range<(x_pos+Tp))[0][-1]
plt.plot(y[:t_max_idx_interaction,Nid]/l0,z[:t_max_idx_interaction,Nid]/l0,"r--",label="t<Tp")

plt.scatter(y[0,Nid]/l0,z[0,Nid]/l0,marker="o",color="k")

max_intensity = plt.Circle((0,0),w0/sqrt(2)/l0,fill=False,ec="red",label="$w_0/\sqrt{2}$")
plt.gca().add_patch(max_intensity)
plt.scatter(0,0,marker="x",color="red")
plt.grid()
plt.legend(loc='lower left',ncol=3)
plt.title("Electron trajectory inside the $F\Theta$ force")
plt.xlabel("$y/\lambda$")
plt.ylabel("$z/\lambda$")
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
    dr =  -(3/8)*a0**2/4*f_squared_prime(r, x_pos)*t**2/2 * (t>2)
    return dr
def dr_Relat(r,t):
    dr = 1/gamma[:,Nid] * -(3/8)*a0**2/4*f_squared_prime(r, x_pos)*t**2/2 * (t>2)
    return dr
# integrate.simpson(dr)

# plt.close("all")
plt.figure()
plt.plot(t_range/l0,r[:,Nid],".-")
# plt.plot(t_range/l0, r[0,Nid]+dr_short_time(r[0,Nid], t_range))
plt.plot(t_range/l0, r[0,Nid]+dr(r[0,Nid],t_range-x_pos),label="r+dr")
plt.plot(t_range/l0, r[0,Nid]+dr_Relat(r[0,Nid],t_range-x_pos),label="r+dr_relat")
plt.legend()
plt.grid()
plt.xlabel("t/t0")

r0 = r[0,Nid]
theta0 = theta[0,Nid]
Lx_integrated_R = integrate.simpson((r0+dr(r0,t_range)) * Ftheta(r0+dr(r0,t_range), theta0, x_pos)*(sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0))**2,x=t_range)
print(Lx_integrated_R)

plt.figure()
temp_env_Z = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
temp_env = sin(pi*(t_range-x_pos)/Tp)**2*((t_range-x_pos)<Tp)*((t_range-x_pos)>0)

plt.plot(t_range/l0,r[:,Nid]*ftheta_time,".-",label="r*Ftheta exact")
plt.plot(t_range/l0,(r0) * Ftheta(r0, theta0, x_pos)*temp_env**2,label="r0*Ftheta0 NR")
plt.plot(t_range/l0,(r0) * Ftheta(r0, theta0, x_pos)*temp_env_Z**2,label="r0*Ftheta0 NR (Tp)")
plt.plot(t_range/l0,(r0+dr(r0,t_range-x_pos)) * Ftheta(r0+dr(r0,t_range-x_pos), theta0, x_pos)*temp_env**2,label="r*Ftheta R")
# plt.plot(t_range/l0,r[:,Nid] * Ftheta(r[:,Nid], theta0, x_pos)*temp_env,label="r*Ftheta R")

plt.grid()
plt.legend()
# @njit
def Lx4_distrib():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = sin(pi*(t_range-x_pos)/Tp)**2*((t_range-x_pos)<Tp)*((t_range-x_pos)>0)
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson((r+dr(r,t_range_lx-x_pos)) * Ftheta(r+dr(r,t_range_lx-x_pos), theta, x_pos)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx_distrib_Relat(use_R=False,use_Z=False,dt=0.1):
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
    Lx_r_range = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            x_model = int(not(use_Z))*x_pos+x[:,Nid]*int(use_Z)
            LxR_distrib = integrate.simpson((r+dr_Relat(r,t_range_lx-x_pos)*use_R) * Ftheta(r+dr_Relat(r,t_range_lx-x_pos)*use_R, theta, x_model)*temp_env**2, x=t_range_lx)

            Lx_theta_list.append(LxR_distrib)
        Lx_r_range.append(np.max(Lx_theta_list))
    return np.array(Lx_r_range)


def Lx_distrib_FullMotion():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_NO_R():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_NO_R_NO_THETA():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[0,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_NO_X():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[0,Nid], x[0,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_GAMMA():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid]*0-x_pos)/Tp)**2*((t_range-x[:,Nid]*0-x_pos)<Tp)*((t_range-x[:,Nid]*0-x_pos)>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x_pos)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_PULSE():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        LxR_distrib = integrate.simpson(r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_NO_GAMMA_NO_PULSE():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x_pos)/Tp)**2*((t_range-x_pos)<Tp)*((t_range-x_pos)>0)
        LxR_distrib = integrate.simpson(r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x_pos)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_Lorentz():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = sin(pi*(t_range-x[:,Nid])/Tp)**2*((t_range-x[:,Nid])<Tp)*((t_range-x[:,Nid])>0)
        Etheta = (y[:,Nid]*Ez[:,Nid] - z[:,Nid]*Ey[:,Nid])/r[:,Nid]
        Br = (y[:,Nid]*By[:,Nid] - z[:,Nid]*Ez[:,Nid])/r[:,Nid]
        pr = (y[:,Nid]*py[:,Nid] - z[:,Nid]*pz[:,Nid])/r[:,Nid]
        gamma = np.sqrt(1+px[:,Nid]**2+py[:,Nid]**2+pz[:,Nid]**2)
        Ftheta_lorentz = -Etheta - px[:,Nid]/gamma*Br + pr/gamma*Bx[:,Nid]

        LxR_distrib = integrate.simpson((Ftheta_lorentz)*temp_env, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

Tp_int_NR = integrate.simpson(temp_env**2,x=t_range)
Tp_int_R = integrate.simpson(temp_env_Z**2,x=t_range)
print("Effective pulse duration:",3/8*Tp/l0,Tp_int_NR/l0,Tp_int_R/l0)

COEF = sqrt(1+a0**2)

plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=2,alpha=0.25)
a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-1])
plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei {a0=}")

Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
plt.plot(r_range/l0,Lx_max_model,"k--",alpha=1)
plt.plot(r_range/l0,-Lx_max_model,"k--",alpha=1, label="Model $\gamma$$L_z^{(2)}$")

d_r_list, d_Lx_list = Lx_distrib_FullMotion_GAMMA()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, lower_Lx,"C1",lw=2)
plt.plot(a_range/l0, upper_Lx,"C1",lw=2,label="Exact integration with $\gamma$, without Tint")

d_r_list, d_Lx_list = Lx_distrib_FullMotion_PULSE()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, lower_Lx,"C2",lw=2)
plt.plot(a_range/l0, upper_Lx,"C2",lw=2,label="Exact integration with Tint, without $\gamma$")

d_r_list, d_Lx_list = Lx_distrib_FullMotion_NO_GAMMA_NO_PULSE()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, lower_Lx,c="purple",lw=2)
plt.plot(a_range/l0, upper_Lx,c="purple",lw=2,label="Exact integration without $\gamma$ and Tint")

d_r_list, d_Lx_list = Lx_distrib_FullMotion()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, lower_Lx,"r",lw=2)
plt.plot(a_range/l0, upper_Lx,"r",lw=2,label="Exact integration")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.title("Compensation of $1/\gamma$ factor and Tint increase")
plt.tight_layout()


plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=2,alpha=0.25)
a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-1])
plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei {a0=}")

Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
plt.plot(r_range/l0,COEF*Lx_max_model,"k--",alpha=1)
plt.plot(r_range/l0,-COEF*Lx_max_model,"k--",alpha=1, label="Model $\gamma$$L_z^{(2)}$")


d_r_list, d_Lx_list = Lx4_distrib()
plt.plot(d_r_list/l0,COEF*d_Lx_list,"C2--",alpha=1)
plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C2--",alpha=1, label="Model $\gamma$$L_z^{(4)}$")

d_r_list, d_Lx_list = Lx_distrib_FullMotion()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, COEF*lower_Lx,"r",lw=2)
plt.plot(a_range/l0, COEF*upper_Lx,"r",lw=2,label="Exact integration *$\gamma$")


# d_r_list, d_Lx_list = Lx_distrib_FullMotion_PULSE()
# plt.scatter(d_r_list/l0, d_Lx_list,s=1,label="Full time integration PULSE")

# d_r_list, d_Lx_list = Lx_distrib_FullMotion_GAMMA()
# plt.scatter(d_r_list/l0, d_Lx_list,s=1,label="Full time integration GAMMA")

# d_r_list, d_Lx_list = Lx_distrib_Lorentz()
# plt.scatter(d_r_list/l0, d_Lx_list,s=10,label="Full time integration LORENTZ")

# d_r_list, d_Lx_list = Lx_distrib_FullMotion_NO_R()
# a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
# plt.plot(a_range/l0, lower_Lx,"r",lw=2)
# plt.plot(a_range/l0, upper_Lx,"r", lw=2,label="Time integration NO R")

# d_r_list, d_Lx_list = Lx_distrib_FullMotion_NO_R_NO_THETA()
# plt.scatter(d_r_list/l0, d_Lx_list,s=1,label="Full time integration NO R NO THETA")

# d_r_list, d_Lx_list = Lx_distrib_FullMotion_NO_X()
# plt.scatter(d_r_list/l0, d_Lx_list,s=1,label="Full time integration NO X")

# Lx_r_range = Lx_distrib(use_R=True)
# plt.plot(r_range/l0,Lx_r_range,"C1")
# plt.plot(r_range/l0,-Lx_r_range,"C1",label="Lx2 time int")

# Lx_max_model_Tint = np.max(LxEpolar(R,THETA,x0,w0,a0,Tp_int_R),axis=0)
# plt.plot(r_range/l0,Lx_max_model_Tint,"C2")
# plt.plot(r_range/l0,-Lx_max_model_Tint,"C2",label="Model Lx2 with eff Tint")

# Lx_r_range_relat = Lx_distrib_Relat(use_R=True)
# plt.plot(r_range/l0,Lx_r_range_relat,"C3--")
# plt.plot(r_range/l0,-Lx_r_range_relat,"C3--",label="Lx2 time int with Relat dr")

# plt.axvline(w0/sqrt(2)/l0,color="r",ls="--")

# Lx_r_range = Lx_distrib(use_R=1)
# plt.plot(r_range/l0,Lx_r_range,"r")
# plt.plot(r_range/l0,-Lx_r_range,"r",label="Lx R time int")
# Lx_r_range = Lx_distrib(use_R=1,use_Z=1)
# plt.plot(r_range/l0,Lx_r_range,"C2")
# plt.plot(r_range/l0,-Lx_r_range,"C2",label="Lx R,Z time int")
plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.title("Lx distribution for different models")
plt.tight_layout()

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