# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:09:46 2024

@author: Jeremy
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

a0_requested = 2.0

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx32",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx32"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6"
# sim_path = "SIM_OPTICAL_A2_HD/opt_a2.0_dx64"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')

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
vx = px/gamma

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


def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3

def Ftheta_V2_O5(r,theta,z):
    numerator = (
        2 * a0**2 * r * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (
            2 * z * np.cos(2 * theta) * (
                4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)
            ) +
            np.sin(2 * theta) * (
                4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * (
                    8 * (w0**4 + 4 * w0**2 + 20) * z**2 +
                    (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4
                ) -
                (w0**4 + 4 * z**2) * (
                    4 * (w0**2 - 2) * z**2 +
                    (w0**2 + 4) * (w0**2 + 6) * w0**2
                )
            )
        )
    )
    denominator = (w0**4 + 4 * z**2)**5
    
    expression = numerator / denominator
    return expression


def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)
def g_gauss(tau):
    return  exp( -((tau-1.25*Tp)/(Tp*3/8/c_gauss))**2) 
def g_sin2(tau):
    return  sin(pi*tau/Tp)**2*(tau>0)*(tau<Tp)
g = g_sin2
def getE(r,theta,z,t):
    tau = t-z
    
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l1)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l1)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l1)-1) * (-2*r**2+w_**2*abs(l1))
    
    Ex = a_*f_*g(tau)*cos(l1*theta + phi_)
    Ey = -eps*a_*f_*g(tau)*sin(l1*theta + phi_)
    Ez = -a_*(f_prime_*g(tau)*cos(theta))*sin(l1*theta + phi_) +\
        a_*f_*g(tau)*(l1/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l1*theta+phi_)
    return Ex,Ey,Ez

ExM, EyM, EzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid],t_range)
AxM, AyM, AzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid]-pi/2,t_range)

Nid = np.where(np.abs(Lx_track[-1])==np.max(np.abs(Lx_track[-1])))[0][0]

Nid = np.where(np.abs(r[0]-1*l0)==np.min(np.abs(r[0]-1*l0)))[0][0]
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
plt.plot(y[:,Nid]/l0,z[:,Nid]/l0,"k",label="r=1*l0 electron trajectory")
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
ftheta_time =FthetaCart(y[:,Nid],z[:,Nid],x[:,Nid])*gauss(t_range,x[:,Nid])**2
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
    delay = 1.25*Tp
    dr =  1*-(3/8)*a0**2/4*f_squared_prime(r, x_pos)*(t-delay)**2/2 * ((t-delay)>0)
    return dr
def dr_Relat(r,t):
    delay = 1.25*Tp+Tp/4
    dr = 1/gamma[:,Nid] * -(3/8)*a0**2/4*f_squared_prime(r, x_pos)*(t-delay)**2/2 * ((t-delay)>0)
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
Lx_integrated_R = integrate.simpson((r0+dr(r0,t_range)) * Ftheta(r0+dr(r0,t_range), theta0, x_pos)*gauss(t_range,x[:,Nid])**2,x=t_range)
print(Lx_integrated_R)

plt.figure()
temp_env_Z = gauss(t_range,x[:,Nid])
temp_env = gauss(t_range,x[:,Nid])

plt.plot(t_range/l0,r[:,Nid]*ftheta_time,".-",label="r*Ftheta exact")
plt.plot(t_range/l0,(r0) * Ftheta(r0, theta0, x_pos)*temp_env**2,label="r0*Ftheta0 NR")
plt.plot(t_range/l0,(r0) * Ftheta(r0, theta0, x_pos)*temp_env_Z**2,label="r0*Ftheta0 NR (Tp)")
plt.plot(t_range/l0,(r0+dr(r0,t_range-x_pos)) * Ftheta(r0+dr(r0,t_range-x_pos), theta0, x_pos)*temp_env**2,label="r*Ftheta R")
# plt.plot(t_range/l0,r[:,Nid] * Ftheta(r[:,Nid], theta0, x_pos)*temp_env,label="r*Ftheta R")

plt.grid()
plt.legend()

azeezaez

# @njit
def Lx4_distrib():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range,x_pos)
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson((r+dr(r,t_range_lx-x_pos)) * Ftheta(r+dr(r,t_range_lx-x_pos), theta, x_pos)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx4_distrib_Relat():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range,x[:,Nid])
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson( 1/gamma[:,Nid]*np.abs(r+dr_Relat(r,t_range_lx-x_pos)) * Ftheta(np.abs(r+dr_Relat(r,t_range_lx-x_pos)), theta, x[:,Nid])*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx4_distrib_FV2():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range,x_pos)
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson((r+dr(r,t_range_lx-x_pos)) * Ftheta_V2_O5(r+dr(r,t_range_lx-x_pos), theta, x_pos)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx4_distrib_Relat_FV2():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range,x[:,Nid])
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson( 1/gamma[:,Nid]*(r+dr_Relat(r,t_range_lx-x_pos)) * Ftheta_V2_O5(r+dr_Relat(r,t_range_lx-x_pos), theta, x[:,Nid])*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_VX():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/(1-vx[:,Nid])*1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_Ax_Px():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/(1-vx[:,Nid])*1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_FV2():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta_V2_O5(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_FV2_O3():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_NO_R():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_NO_R_NO_THETA():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[0,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_NO_X():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[0,Nid] * Ftheta(r[0,Nid], theta[0,Nid], x[0,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_GAMMA():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x_pos) #sin(pi*(t_range-x[:,Nid]*0-x_pos)/Tp)**2*((t_range-x[:,Nid]*0-x_pos)<Tp)*((t_range-x[:,Nid]*0-x_pos)>0)
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x_pos)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_PULSE():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_NO_GAMMA_NO_PULSE():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x_pos)
        LxR_distrib = integrate.simpson(r[:,Nid] * Ftheta(r[:,Nid], theta[:,Nid], x_pos)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx_distrib_Lorentz():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
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


COEF = sqrt(1+(a0)**2)
plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=2,alpha=0.25)
a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-1])
plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei {a0=}")

Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x_pos))**2+ 1/4*(a0*f(r_range,x_pos))**4)
plt.plot(r_range/l0,COEF*Lx_max_model,"k--",alpha=1)
plt.plot(r_range/l0,-COEF*Lx_max_model,"k--",alpha=1, label="Model $\gamma$$L_z^{(2)}$")


d_r_list, d_Lx_list = Lx4_distrib()
COEF = sqrt(1+(a0*f(d_r_list,x_pos))**2+ 1/4*(a0*f(d_r_list,x_pos))**4)
plt.plot(d_r_list/l0,COEF*d_Lx_list,"C2--",alpha=1)
plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C2--",alpha=1, label="Model $\gamma$$L_z^{(4)}$")


d_r_list, d_Lx_list = Lx4_distrib_Relat()
COEF = sqrt(1+(a0*f(d_r_list,x_pos))**2+ 1/4*(a0*f(d_r_list,x_pos))**4)
plt.plot(d_r_list/l0,COEF*d_Lx_list,"C3--",alpha=1)
plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C3--",alpha=1, label="Model $\gamma$$L_z^{(4)}$ Relat")

d_r_list, d_Lx_list = Lx_distrib_FullMotion()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
COEF = sqrt(1+(a0*f(a_range,x_pos))**2+ 1/4*(a0*f(a_range,x_pos))**4)
plt.plot(a_range/l0, COEF*lower_Lx,"r-",lw=2)
plt.plot(a_range/l0, COEF*upper_Lx,"r-",lw=2,label="Exact integration *$\gamma$")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.title("Lx distribution for different models")
plt.tight_layout()








#===============================================
# USING UPDATE F_THETA
#===============================================
COEF = sqrt(1+(a0)**2)
plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=2,alpha=0.25)
a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-1])
plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei {a0=}")

Lx_max_model = np.max(LxEpolar(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x_pos))**2+ 1/4*(a0*f(r_range,x_pos))**4)
plt.plot(r_range/l0,COEF*Lx_max_model,"k--",alpha=1)
plt.plot(r_range/l0,-COEF*Lx_max_model,"k--",alpha=1, label="Model $\gamma$$L_z^{(2)}$")

d_r_list, d_Lx_list = Lx4_distrib_Relat_FV2()
COEF = sqrt(1+(a0*f(d_r_list,x_pos))**2+ 1/4*(a0*f(d_r_list,x_pos))**4)
plt.plot(d_r_list/l0,COEF*d_Lx_list,"k--",alpha=1)
plt.plot(d_r_list/l0,-COEF*d_Lx_list,"k--",alpha=1, label="Model $L_z^{(4)}$ * $\sqrt{1+(a_0f(r))^2+1/4(a_0f(r))^4}$")

d_r_list, d_Lx_list = Lx_distrib_FullMotion()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
COEF = sqrt(1+(a0*f(a_range,x_pos))**2+ 1/4*(a0*f(a_range,x_pos))**4)
plt.plot(a_range/l0, COEF*lower_Lx,"r-",lw=2)
plt.plot(a_range/l0, COEF*upper_Lx,"r-",lw=2,label="Exact integration *$\sqrt{1+(a_0f(r))^2+1/4(a_0f(r))^4}$")


d_r_list, d_Lx_list = Lx_distrib_FullMotion_VX()
a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
plt.plot(a_range/l0, lower_Lx,"-.",color="purple",lw=2)
plt.plot(a_range/l0, upper_Lx,"-.",color="purple",lw=2,label="Exact integration *$1/(1-v_x)$")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.title("Lx distribution for different models")
plt.tight_layout()






# plt.figure()
# temp_env = gauss(t_range,x[:,Nid])
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