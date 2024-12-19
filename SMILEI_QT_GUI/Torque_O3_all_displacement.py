# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:52:41 2024

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
import time
plt.close("all")

a0_requested = 2

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx48"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6"
# sim_path = "SIM_OPTICAL_A2_HD/opt_a2.0_dx64"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/gauss_a2_Tp6_NET_GAIN_dx64')

l0=2*np.pi
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1


track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range_smooth = np.arange(0,t_range[-1],0.5)
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1
# r_range_net = S.namelist.r_range_net

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
gamma = sqrt(1+px**2+py**2+pz**2)
vx = px/gamma


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

def averageAM(X,Y,dr_av):
    da = 0.04
    t0 = time.perf_counter()
    print("Computing average...",da)
    a_range = np.arange(0,np.max(X)*1.0+da,da)
    av_Lx = np.empty(a_range.shape)
    std_Lx = np.empty(a_range.shape)
    for i,a in enumerate(a_range):
        mask = (X > a-dr_av/2) & (X < a+dr_av/2)
        av_Lx[i] = np.nanmean(Y[mask])
        std_Lx[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
    t1 = time.perf_counter()
    print(f"...{(t1-t0):.0f} s")
    return a_range,av_Lx, std_Lx
# def averageModified(X,Y,dr_av):
#     t0 = time.perf_counter()
#     print("Computing average...")
#     a_range = r_range_net
#     M = np.empty(a_range.shape)
#     STD = np.empty(a_range.shape)
#     for i,a in enumerate(a_range):
#         mask = (X > a-dr_av/2) & (X < a+dr_av/2)
#         M[i] = np.nanmean(Y[mask])
#         STD[i] = np.std(Y[mask])/np.sqrt(len(Y[mask]))
#     t1 = time.perf_counter()
#     print(f"...{(t1-t0):.0f} s")
#     return a_range,M, STD

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
def Torque_V2_O3(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r**2 * w0**6 * 
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




def gauss_squared_prime(t,x):
    t_center = 1.25*Tp
    c_gauss = sqrt(pi/2)
    sigma = 3/8*Tp/c_gauss
    return 4/sigma**2 * (t-t_center-x) * gauss(t,x)**2

zR = 0.5*w0**2
eps,l1 = 0,1
C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)

def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)
g = gauss
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

t_center=1.25*Tp
c_gauss = sqrt(pi/2)
sigma_gauss = Tp*3/8/c_gauss


def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x_pos)/sigma_gauss)+1)
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

#==================================================
r_requested = 1.*l0
Nid = np.where(np.abs(r[0]-r_requested)==np.min(np.abs(r[0]-r_requested)))[0][0]
#==================================================

x_pos = 5*l0
x0 = x_pos
r0 = r[0,Nid]
theta0 = theta[0,Nid]
x_range = np.arange(-1*w0,1*w0,0.1)
X,Y = np.meshgrid(x_range,x_range)
extent = [x_range[0]/l0,x_range[-1]/l0,x_range[0]/l0,x_range[-1]/l0]

# plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=1)

r_range = np.arange(0,4*l0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R,THETA = np.meshgrid(r_range,theta_range)
Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,x0,w0,a0,3/8*Tp),axis=0)

def Lx4_distrib_max(dr_func, dtheta_func, dx_func):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    distrib_Lx_list = []
    for r0 in r_range:
        Lx_theta_list = []
        for theta0 in np.arange(0,2*pi,pi/16):
            LxR_distrib = integrate.simpson(Torque_V2_O3(np.abs(r0+dr_func(r0,t_range_lx,x0)), theta0+dtheta_func(r0,theta0,t_range_lx,x0), x0+dx_func(r0,t_range_lx,x0))*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)
# @njit
def Lx4_distrib_mean(dr_func, dtheta_func, dx_func, N=10_000):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    mean_Lx_list = []
    std_Lx_list = []
    r_range = np.linspace(0.1,4*l0,25)
    for r0 in r_range:
        t0 = time.perf_counter()
        Lx_theta_list = []
        for theta0 in np.linspace(0,2*np.pi,N):
            LxR_distrib = integrate.simpson(Torque_V2_O3(np.abs(r0+dr_func(r0,t_range_lx,x0)), theta0+dtheta_func(r0,theta0,t_range_lx,x0), x0+dx_func(r0,t_range_lx,x0))*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        mean_Lx_list.append(np.nanmean(Lx_theta_list))
        # print(sqrt(len(Lx_theta_list)))
        std_Lx_list.append(np.std(Lx_theta_list)/sqrt(len(Lx_theta_list)))
        t1 = time.perf_counter()
        print(f"{r0/r_range[-1]*100:.0f}% - time theta av: {(t1-t0):.2f}s (sqrt(N)={sqrt(len(Lx_theta_list)):.2f})")
    return np.array(r_range),np.array(mean_Lx_list), np.array(std_Lx_list)

def Lx_distrib_FullMotion():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * r[:,Nid] * Ftheta_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)





def dx_func(r,t,x):
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return a0**2/4 * f(r,x)**2 * gauss2_int(t,x)

def dtheta_func(r,theta,t,x):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return 1/gamma*Ftheta_V2_O3(r,theta,x) * gauss2_int_int(t, x)
    

plt.figure()
plt.plot(t_range/l0, theta[:,Nid],".-")
plt.plot(t_range_smooth/l0, theta0 + dtheta_func(r0,theta0,t_range_smooth,x0))
plt.grid()




# plt.figure()
# # plt.scatter(r[0]/l0,Lx_track[-1],s=4,alpha=1)
# a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-2])
# plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
# plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
# plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei {a0=}")

# Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
# COEF = sqrt(1+(a0*f(r_range,x_pos))**2+ 1/4*(a0*f(r_range,x_pos))**4)
# plt.plot(r_range/l0,COEF*Lx_max_model,"k--")
# plt.plot(r_range/l0,-COEF*Lx_max_model,"k--", label="Model $\gamma$$L_z^{NR}$")

# d_r_list, d_Lx_list = Lx4_distrib_max(dr_Relat_mean,lambda r,theta,t,x: 0, lambda r,t,x: 0)
# COEF = sqrt(1+(f(d_r_list+dr(d_r_list,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(d_r_list+dr(d_r_list,1.25*Tp+x0,x0),5*l0)*a0)**4)
# plt.plot(d_r_list/l0,COEF*d_Lx_list,"C1-",lw=2)
# plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C1-",lw=2, label="Model $\gamma_{max}$$L_z^{R}$")

# d_r_list, d_Lx_list = Lx4_distrib_max(dr_Relat_mean,lambda r,theta,t,x: 0, dx_func)
# COEF = sqrt(1+(f(d_r_list+dr(d_r_list,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(d_r_list+dr(d_r_list,1.25*Tp+x0,x0),5*l0)*a0)**4)
# plt.plot(d_r_list/l0,COEF*d_Lx_list,"C2-",lw=2)
# plt.plot(d_r_list/l0,-COEF*d_Lx_list,"C2-",lw=2, label="Model $\gamma_{max}$$L_z^{R}$ with dx")

# plt.grid()
# plt.legend()
# plt.xlabel("$r_0/\lambda$")
# plt.ylabel("$L_x$")
# plt.title(f"Lx distribution for different models\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
# plt.tight_layout()



wolframe_table_path = r"C:\Users\Jeremy\_LULI_\_STAGE_LULI_\MATHEMATICA_TABLES\table_LG_LZ4_mymodel_w2.5_a2.332.txt"
table = np.loadtxt(wolframe_table_path)
y_range_table = np.arange(-1.75*w0, 1.75*w0+l0/128, l0/128)
Y_table, Z_table = np.meshgrid(y_range_table,y_range_table)
R_table = sqrt(Y_table**2+Z_table**2).ravel()
a_range_table, av_Lx_table, std_Lx_table = averageAM(R_table, table, 0.5)


plt.figure()
# plt.plot(a_range_table/l0,av_Lx_table,"k--")

# a_range, mean_Lx, std_Lx = averageModified(r[0],Lx_track[-1],dr_av = 0.1)
# plt.plot(a_range/l0, mean_Lx,".-",label=f"Smilei dx=$\lambda$/{l0/S.namelist.dx:.0f}")
# plt.fill_between(a_range/l0, mean_Lx-std_Lx, mean_Lx+std_Lx,alpha=0.25)


r_range0, mean_Lx0, std_Lx0 = Lx4_distrib_mean(lambda r,t,x:0,
                                               lambda r,theta,t,x: 0, 
                                               lambda r,t,x:0, N=200_000)
COEF0 = sqrt(1+(f(r_range0+dr(r_range0,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(r_range0+dr(r_range0,1.25*Tp+x0,x0),5*l0)*a0)**4)


r_range1, mean_Lx1, std_Lx1 = Lx4_distrib_mean(lambda r,t,x:0,
                                               lambda r,theta,t,x: 0, 
                                               dx_func, N=200_000)
COEF1 = sqrt(1+(f(r_range1+dr(r_range1,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(r_range1+dr(r_range1,1.25*Tp+x0,x0),5*l0)*a0)**4)


r_range2, mean_Lx2, std_Lx2 = Lx4_distrib_mean(lambda r,t,x:0,
                                               dtheta_func, 
                                               lambda r,t,x: 0, N=200_000)
COEF2 = sqrt(1+(f(r_range2+dr(r_range2,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(r_range2+dr(r_range2,1.25*Tp+x0,x0),5*l0)*a0)**4)


r_range3, mean_Lx3, std_Lx3 = Lx4_distrib_mean(lambda r,t,x:0,
                                               dtheta_func, 
                                               dx_func, N=200_000)
COEF3 = sqrt(1+(f(r_range3+dr(r_range3,1.25*Tp+x0,5*l0),x0)*a0)**2+ 1/4*(f(r_range3+dr(r_range3,1.25*Tp+x0,x0),5*l0)*a0)**4)

plt.plot(r_range1/l0,mean_Lx0,"k-",lw=2,label="Model <$L_x$> (0, 0, 0)")
plt.plot(r_range1/l0,mean_Lx1,"C0-",lw=2,label="Model <$L_x$> (0, 0, $\Delta x$)")
plt.plot(r_range3/l0,mean_Lx2,"C2-",lw=2,label="Model <$L_x$> (0, $\Delta \Theta$, $\Delta x$)")
plt.plot(r_range3/l0,mean_Lx3,"C3-",lw=2,label="Model <$L_x$> (0, $\Delta \Theta$, $\Delta x$)")


# plt.fill_between(r_range3/l0, COEF*(mean_Lx3-std_Lx3/2), COEF*(mean_Lx3+std_Lx3/2), alpha=0.25, color="C2")
# plt.fill_between(r_range3/l0, COEF*(mean_Lx3-std_Lx3/2), COEF*(mean_Lx3+std_Lx3/2), alpha=0.25, color="C2")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("<$L_x$>")
plt.title(f"Lx distribution for different models\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
plt.tight_layout()



