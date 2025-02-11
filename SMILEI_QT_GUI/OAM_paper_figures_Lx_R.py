# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:04:25 2025

@author: Jeremy
"""


import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
from tqdm import tqdm
plt.close("all")

l0=2*np.pi
x0=5*l0
w0=2.5*l0
Tp=12*l0
sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM4_R8"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

max_time = [-1,
            -1,
            -1,
            -1,
            -2,
            -3]



def min_max(X,Y,dr_av=0.15):
    if len(X) < 100_000: 
        print("Low N for min_max: siwtcheted to dr_av =0.5")
        dr_av=0.5
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

def LxEpolar_V2_O3(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 +
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) -
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 +
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) +
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr
def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O3(r,theta,z)

def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O5(r,theta,z)
def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3
# @njit
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
                    (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * (4 * (w0**2 - 2) * z**2 +(w0**2 + 4) * (w0**2 + 6) * w0**2))))
    denominator = (w0**4 + 4 * z**2)**5
    expression = numerator / denominator
    return expression
def Torque_V2_O3(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r**2 * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3
def Torque_V2_O5(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    numerator = ( 2 * a0**2 * r**2 * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (2 * z * np.cos(2 * theta) * ( 4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)) +
            np.sin(2 * theta) * (  4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * ( 8 * (w0**4 + 4 * w0**2 + 20) * z**2 + (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * ( 4 * (w0**2 - 2) * z**2 + (w0**2 + 4) * (w0**2 + 6) * w0**2  ))))
    denominator = (w0**4 + 4 * z**2)**5
    return numerator / denominator

def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)
t_center=1.25*Tp
sigma_gauss = Tp*3/8/c_gauss
 


def g_gauss(tau):
    return  exp( -((tau-1.25*Tp)/(Tp*3/8/c_gauss))**2)
def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x0)/sigma_gauss)+1)
def gauss2_int_int(t,x):
    t_center = 1.25*Tp
    psi = lambda t,x : sqrt(2)*(t-t_center-x)/sigma_gauss
    expr =lambda t,x : 0.5*sqrt(pi/2)*sigma_gauss*( gauss(t,x)**2*sigma_gauss/sqrt(2*pi) + t + (t-t_center-x)*(special.erf(psi(t,x))))
    return expr(t,x) - expr(0,x)

def dr_Relat_mean(r,t,x):
    return -1/(1+(f(r,x)*a0)**2/2+ 1/4*1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_Relat(r,t,x):
    return -1/(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr(r,t,x):
    return -1/sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)



def Lx4_distrib(dr_func):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    distrib_Lx_list = []
    for r in tqdm(r_range):
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(r+dr_func(r,t_range_lx,x0)), theta, x0)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx4_distrib_GAMMA(dr_func):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    distrib_Lx_list = []
    for r in tqdm(r_range):
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/32):
            COEF_time = sqrt(1+(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**2+ 1/4*(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**4)
            LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(np.abs(r+dr(r,t_range_lx,x0)), theta, x0)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)


r_range = np.arange(0,2*w0,0.02)
theta_range = np.arange(0,2*pi,pi/32)
R,THETA = np.meshgrid(r_range,theta_range)


fig, axs = plt.subplots(2,len(sim_loc_list_12)//2, figsize=(10,5))


for k,sim_path in enumerate(sim_loc_list_12[:]):

    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
    
    ax = axs.ravel()[k]
    print(k)
    
    
    T0 = S.TrackParticles("track_eon_full", axes=["x","y","z","py","pz","px"])
    
    Ltrans = S.namelist.Ltrans
    Tp = S.namelist.Tp
    w0 = S.namelist.w0
    a0 = S.namelist.a0
    l1 = S.namelist.l1
    dx = S.namelist.dx
    
    track_N_tot = T0.nParticles
    t_range = T0.getTimes()
    t_range_smooth = np.arange(0,t_range[-1],0.01)
    track_traj = T0.getData()
    
    print(f"RESOLUTION: l0/{l0/S.namelist.dx}")
    
    N_part = 1
    
    if "AM" in sim_path:
        dx_string = f"dx={l0/dx:.0f} AM={S.namelist.number_AM_modes}"
    
        track_r_center = 0
    else:
        dx_string = f"dx={l0/dx:.0f}"
        track_r_center = Ltrans/2
    
    x = track_traj["x"][:,::N_part]
    y = track_traj["y"][:,::N_part]-track_r_center
    z = track_traj["z"][:,::N_part] -track_r_center
    py = track_traj["py"][:,::N_part]
    pz = track_traj["pz"][:,::N_part]
    px = track_traj["px"][:,::N_part]
    
    
    r = np.sqrt(y**2+z**2)
    theta = np.arctan2(z,y)
    pr = (y*py + z*pz)/r
    Lx_track =  y*pz - z*py
    gamma = sqrt(1+px**2+py**2+pz**2)
    vx = px/gamma
    
    tmax = 120
    

    a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[max_time[k]])
    ax.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
    ax.plot(a_range/l0,lower_Lx,"C0",lw=2)
    ax.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei")
    
    
    Lx_max_model = np.max(LxEpolar_V2_O5(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
    COEF = sqrt(1+(a0*f(r_range,x0))**2+ 1/4*(a0*f(r_range,x0))**4)
    ax.plot(r_range/l0,COEF*Lx_max_model,"k-", lw=2)
    ax.plot(r_range/l0,-COEF*Lx_max_model,"k-", lw=2, label="Model $\gamma$$L_x^{NR}$")
    
    # r_range_Lx, Lx_distrib = Lx4_distrib(dr_Relat_mean)
    # ax.plot(r_range/l0,COEF*Lx_distrib,"C1-", lw=2)
    # ax.plot(r_range/l0,-COEF*Lx_distrib,"C1-", lw=2, label="Model $\gamma$$L_x^{R}$")
    
    r_range_Lx, Lx_distrib = Lx4_distrib_GAMMA(dr_Relat_mean)
    ax.plot(r_range/l0,Lx_distrib,"C1-", lw=2)
    ax.plot(r_range/l0,-Lx_distrib,"C1-", lw=2, label="Model $\gamma$$L_x^{R}$")
    
    if k==0: ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    ax.legend(ncol=3,columnspacing=0.7,handlelength=1.5,fontsize=12)
    ax.grid()
    ax.set_title(f"$\mathbf{{a_0={a0}}}$",fontsize=16)
    plt.pause(0.1)


axs.ravel()[0].set_ylabel("$L_x$",fontsize=16)
axs.ravel()[3].set_ylabel("$L_x$",fontsize=16)

axs.ravel()[3].set_xlabel("$r_0/\lambda$",fontsize=16)
axs.ravel()[4].set_xlabel("$r_0/\lambda$",fontsize=16)
axs.ravel()[5].set_xlabel("$r_0/\lambda$",fontsize=16)


for i,ax in enumerate(axs.ravel()):
    handles, previous_labels = ax.get_legend_handles_labels()
    new_labels = ["Smilei", "$\gammaL_x^{NR}$","$\gammaL_x^{R}$"]
    ax.legend(loc="lower center",handles=handles, labels=new_labels,ncol=3, 
              columnspacing=0.6,handlelength=1.4,fontsize=12)
    if i!=0: ax.get_legend().remove()

fig.tight_layout()
    



# d_r_list, d_Lx_list = Lx4_distrib()
# COEF = sqrt(1+(a0*f(d_r_list,x0))**2+ 1/4*(a0*f(d_r_list,x0))**4)
# plt.plot(d_r_list/l0,COEF*d_Lx_list,"C1-",lw=2)
# plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C1-",lw=2, label="Model $\gamma$$L_z^{(4)}$")


# d_r_list, d_Lx_list = Lx4_distrib(dr_Relat_mean)
# COEF = sqrt(1+(f(d_r_list+dr(d_r_list,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(d_r_list+dr(d_r_list,1.25*Tp+x0,x0),5*l0)*a0)**4)
# COEF = sqrt(1+(f(d_r_list,5*l0)*a0)**2+ 1/4*(f(d_r_list,5*l0)*a0)**4)

# plt.plot(d_r_list/l0,COEF*d_Lx_list,"C1-",lw=2)
# plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C1-",lw=2, label="Model $\gamma_{max}$$L_z^{R}$")


# d_r_list, d_Lx_list = Lx_distrib_FullMotion()
# a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list)
# COEF = sqrt(1+(f(a_range+dr(a_range,20*l0,5*l0),5*l0)*a0)**2+ 1/4*(f(a_range+dr(a_range,20*l0,5*l0),5*l0)*a0)**4)
# plt.plot(a_range/l0,COEF*lower_Lx,"r-",lw=2)
# plt.plot(a_range/l0,COEF*upper_Lx,"r-",lw=2, label="Exact integration")

