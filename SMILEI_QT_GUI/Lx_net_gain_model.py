# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:59:20 2025

@author: Jeremy
"""



import os
import sys
import numpy as np
from numpy import exp, sin, cos, arctan2, pi, sqrt

import matplotlib.pyplot as plt
module_dir_happi = f"{os.environ['SMILEI_SRC']}"
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import time
from tqdm import tqdm
plt.close("all")

a0_requested = 4

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,4])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]

# sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp6"
# sim_path = "SIM_OPTICAL_A2_HD/opt_a2.0_dx64"

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp6')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_NET_GAIN/gauss_a3_Tp12_NET_GAIN_dx128_AM4')

l0=2*np.pi
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1
# r_range_net = S.namelist.r_range_net


track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range_smooth = np.arange(0,t_range[-1],0.5)
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1
# r_range_net = S.namelist.r_range_net
if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2
x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part] - track_r_center
z = track_traj["z"][:,::N_part] - track_r_center
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


def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*Torque_V2_O3(r,theta,z)

def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
    return Tint*Torque_V2_O5(r,theta,z)

# @njit
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

def Ftheta_V2_O3(r,theta,X):
    return Torque_V2_O3(r,theta,X)/r
def Ftheta_V2_O5(r,theta,X):
    return Torque_V2_O5(r,theta,X)/r

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
def getE(r,theta,z,t):
    tau = t-z
    g_ = 1#gauss(t,z)

    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l1)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l1)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l1)-1) * (-2*r**2+w_**2*abs(l1))
    
    Ex = a_*f_*g_*cos(l1*theta + phi_)
    Ey = -eps*a_*f_*g_*sin(l1*theta + phi_)
    Ez = -a_*(f_prime_*g_*cos(theta))*sin(l1*theta + phi_) +\
        a_*f_*g_*(l1/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l1*theta+phi_)
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

x_pos = 5*l0
x0 = x_pos
x_range = np.arange(-1*w0,1*w0,0.1)
X,Y = np.meshgrid(x_range,x_range)
extent = [x_range[0]/l0,x_range[-1]/l0,x_range[0]/l0,x_range[-1]/l0]

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
    t_range_smooth = np.arange(0,t_range[-1],1)
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    mean_Lx_list = []
    std_Lx_list = []
    r_range = np.linspace(0.8*l0,1.2*w0/sqrt(2),25)
    for r0 in tqdm(r_range):
        # t0 = time.perf_counter()
        Lx_theta_list = []
        for theta0 in np.linspace(0,2*np.pi,N):
            LxR_distrib = integrate.simpson(Torque_V2_O3(np.abs(r0+dr_func(r0,t_range_lx,x0)), theta0+dtheta_func(r0,theta0,t_range_lx,x0), x0+dx_func(r0,t_range_lx,x0))*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        mean_Lx_list.append(np.nanmean(Lx_theta_list))
        # print(sqrt(len(Lx_theta_list)))
        std_Lx_list.append(np.std(Lx_theta_list)/sqrt(len(Lx_theta_list)))
        # t1 = time.perf_counter()
        # print(f"{r0/r_range[-1]*100:.0f}% - time theta av: {(t1-t0):.2f}s (sqrt(N)={sqrt(len(Lx_theta_list)):.2f})")
    return np.array(r_range),np.array(mean_Lx_list), np.array(std_Lx_list)


def Lx4_distrib_GAMMA(dr_func):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    distrib_Lx_list = []
    for r in tqdm(r_range):
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/32):
            COEF_time = sqrt(1+(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**2+ 1/4*(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**4)

            LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(np.abs(r+dr(r,t_range_lx,x0)), theta, x_pos)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_CONTROL_FV2O5(coord):
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        rp, thetap, xp = r[0,Nid], theta[0,Nid],  x[0,Nid]
        if "x" in coord:
            xp = x[:,Nid]
        if "theta" in coord:
            thetap = theta[:,Nid]    
        if "r" in coord:
            rp = r[:,Nid]
        temp_env = gauss(t_range,x[0,Nid])
        LxR_distrib = integrate.simpson(Torque_V2_O5(rp, thetap, xp)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def Lx4_distrib_mean(coord, N=50_000):
    t_range_smooth = np.arange(0,t_range[-1],0.1)
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    mean_Lx_list = []
    std_Lx_list = []
    r_range = np.linspace(0.5*l0,2*l0,25)
    dr, dtheta, dx = lambda r,theta,x,t:0,lambda r,theta,x,t:0,lambda r,theta,x,t:0

    if "x" in coord:
        dx = dx_func
    if "theta" in coord:
        dtheta = dtheta_func_NT    
    if "r" in coord:
        dr = dr_func
    
    for r0 in tqdm(r_range):
        # t0 = time.perf_counter()
        Lx_theta_list = []
        for theta0 in np.linspace(0,2*np.pi,N):
            LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(r0+dr(r0,theta0,x0,t_range_lx)), 
                                                         theta0+dtheta(r0,theta0,x0,t_range_lx), 
                                                         x0+dx(r0,theta0,x0,t_range_lx))*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        mean_Lx_list.append(np.nanmean(Lx_theta_list))
        # print(sqrt(len(Lx_theta_list)))
        std_Lx_list.append(np.std(Lx_theta_list)/sqrt(len(Lx_theta_list)))
        # t1 = time.perf_counter()
        # print(f"{r0/r_range[-1]*100:.0f}% - time theta av: {(t1-t0):.2f}s (sqrt(N)={sqrt(len(Lx_theta_list)):.2f})")
    return np.array(r_range),np.array(mean_Lx_list), np.array(std_Lx_list)



def dx_func(r,theta,x,t):
    """ use 1/gamma ?"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return a0**2/4 * f(r,x)**2 * gauss2_int(t,x) #/gamma

def dtheta_func(r,theta,x,t):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return 1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)

def dtheta_func_NT(r0,theta0,x0,t):
    r_model = np.abs(r0 + dr_func(r0,theta0,x0, t))
    x_model = x0 +dx_func(r0, theta0, x0, t)

    idx_cross = np.where(r_model==np.min(r_model))[0][0]
    theta0_m = np.zeros_like(r_model) + theta0
    theta0_m[idx_cross:] = pi+theta0_m[idx_cross:] #Allow model to cross beam axis and switch theta

    y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(t,x_model)*cos(l1*theta0_m  - t + x_model)
    z_model = r_model*sin(theta0_m)
    theta_model = np.arctan2(z_model, y_model) + dtheta_func(r0, theta0_m, x_model, t)
    return theta_model - theta0


def dr_func(r,theta,x,t):
    return dr_Relat_mean(r, t, x)



"""
r_range3, mean_Lx3, std_Lx3 = Lx4_distrib_mean(lambda r,t,x:0,
                                                dtheta_func, 
                                                dx_func, N=50_000)
COEF3 = sqrt(1+(f(r_range3,x0)*a0)**2+ 1/4*(f(r_range3,x0)*a0)**4)
"""

SAVE = False
# LOAD = False


# r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean("",N=50_000)
# COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
# plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "k--",label="None")
# plt.fill_between(COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="k",alpha=0.2)
# plt.pause(0.1)

# r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean("r",N=50_000)
# COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
# plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C0--",label="r")
# plt.fill_between(COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C0",alpha=0.2)
# plt.pause(0.1)

# r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean("theta",N=50_000)
# COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
# plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C1--",label="theta")
# plt.fill_between(COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C1",alpha=0.2)
# plt.pause(0.1)

# r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean("x",N=50_000)
# COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
# plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C2--",label="x")
# plt.fill_between(COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C2",alpha=0.2)
# plt.pause(0.1)

# r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean("r x",N=50_000)
# COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
# plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C3-",label="r x")
# plt.fill_between(COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C3",alpha=0.2)
# plt.pause(0.1)

coords_var = "r_theta"
N = 10_000
try:
    aez
    data = np.loadtxt(rf"{os.environ['SMILEI_QT']}\data\net_gain_model\net_gain_model_{coords_var}_{N}_a{a0:.1f}.txt")
    r_range_Model, mean_Lx_Model, std_Lx_Model = data[:,0], data[:,1], data[:,2]
except:
    r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean(coords_var,N=N)
    if SAVE:  np.savetxt(rf"{os.environ['SMILEI_QT']}\data\net_gain_model\net_gain_model_{coords_var}_{N}_a{a0}.txt", np.column_stack((r_range_Model, mean_Lx_Model,std_Lx_Model)))

COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C1-",label=coords_var)
# plt.fill_between(r_range_Model/l0,COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C4",alpha=0.1)
plt.pause(0.1)

coords_var = "theta_x"
N = 10_000
try:
    aze
    data = np.loadtxt(rf"{os.environ['SMILEI_QT']}\data\net_gain_model\net_gain_model_{coords_var}_{N}_a{a0:.1f}.txt")
    r_range_Model, mean_Lx_Model, std_Lx_Model = data[:,0], data[:,1], data[:,2]
except:
    r_range_Model, mean_Lx_Model, std_Lx_Model = Lx4_distrib_mean(coords_var,N=N)
    if SAVE:  np.savetxt(rf"{os.environ['SMILEI_QT']}\data\net_gain_model\net_gain_model_{coords_var}_{N}_a{a0:.1f}.txt", np.column_stack((r_range_Model, mean_Lx_Model,std_Lx_Model)))

COEF_Model = sqrt(1+(a0*f(r_range_Model,x0))**2+ 1/4*(a0*f(r_range_Model,x0))**4)
plt.plot(r_range_Model/l0, COEF_Model*mean_Lx_Model, "C2-",label=coords_var)
# plt.fill_between(r_range_Model/l0,COEF_Model*(mean_Lx_Model-2*std_Lx_Model),COEF_Model*(mean_Lx_Model+2*std_Lx_Model),color="C5",alpha=0.1)
plt.pause(0.1)
plt.grid()
plt.legend()
plt.title("Net gain $<L_x^R>$ model")
plt.tight_layout()

eazaezaez


def Lx4_distrib_GAMMA(coord):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    r_range = np.arange(0.*w0,1.75*w0,0.01)
    max_Lx = []
    min_Lx = []
    
    dr, dtheta, dx = lambda r,theta,x,t:0,lambda r,theta,x,t:0,lambda r,theta,x,t:0
    if "x" in coord:
        dx = dx_func
    if "theta" in coord:
        dtheta = dtheta_func    
    if "r" in coord:
        dr = dr_func
    

    for r0 in tqdm(r_range):
        Lx_theta_list = []
        for theta0 in np.arange(0,2*pi,pi/64):
            COEF_time = sqrt(1+(f(r0+dr_Relat_mean(r0,t_range_lx,x0),x0)*a0)**2+ 1/4*(f(r0+dr_Relat_mean(r0,t_range_lx,x0),x0)*a0)**4)

            LxR_distrib = integrate.simpson(Torque_V2_O3(np.abs(r0+dr(r0,theta0,x0,t_range_lx)), theta0+dtheta(r0,theta0,x0,t_range_lx),x0+dx(r0,theta0,x0,t_range_lx))*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        max_Lx.append(np.max(Lx_theta_list))
        min_Lx.append(np.min(Lx_theta_list))
    return np.array(r_range),np.array(min_Lx),np.array(max_Lx)
plt.figure()
coords_var = "theta_x"
d_r_list, min_Lx_model, max_Lx_model = Lx4_distrib_GAMMA(coords_var)
plt.plot(d_r_list/l0,min_Lx_model,"C1--",lw=2)
plt.plot(d_r_list/l0,max_Lx_model,"C1--",lw=2, label=coords_var)
mean_max_modelS = np.mean([min_Lx_model,max_Lx_model],axis=0)
plt.plot(d_r_list/l0,mean_max_modelS,"C1")
coords_var = "theta_r"
d_r_list, min_Lx_model, max_Lx_model = Lx4_distrib_GAMMA(coords_var)
plt.plot(d_r_list/l0,min_Lx_model,"C2--",lw=2)
plt.plot(d_r_list/l0,max_Lx_model,"C2--",lw=2, label=coords_var)
mean_max_modelS = np.mean([min_Lx_model,max_Lx_model],axis=0)
plt.plot(d_r_list/l0,mean_max_modelS,"C2")
coords_var = "r_theta_x"
d_r_list, min_Lx_model, max_Lx_model = Lx4_distrib_GAMMA(coords_var)
plt.plot(d_r_list/l0,min_Lx_model,"C3--",lw=2)
plt.plot(d_r_list/l0,max_Lx_model,"C3--",lw=2, label=coords_var)
mean_max_modelS = np.mean([min_Lx_model,max_Lx_model],axis=0)
plt.plot(d_r_list/l0,mean_max_modelS,"C3")


plt.xlabel("$r_0/\lambda$")
plt.title(f"Net gain $<L_x>$ from model ($a_0={a0}$, $T_p={Tp/l0:.2f}\lambda/c$)")

plt.axhline(0,color="k",ls="--")
plt.axvline(1,color="k",ls="--")
plt.axvline(1.8,color="k",ls="--")
plt.xlim(0,2)
plt.grid()
plt.legend()

azeazeazeza

def getE(r,theta,z,t):
    
    tau = t-z
    g_ = 1

    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l1)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l1)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l1)-1) * (-2*r**2+w_**2*abs(l1))

    Ex = a_*f_*g_*cos(l1*theta + phi_)
    Ey = -eps*a_*f_*g_*sin(l1*theta + phi_)
    Ez = -a_*(f_prime_*g_*cos(theta))*sin(l1*theta + phi_) +\
        a_*f_*g_*(l1/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l1*theta+phi_)
    return Ex,Ey,Ez
theta_range = np.arange(0,2*pi,pi/64)
r_range = np.arange(0,2*w0,0.1)
x_range = np.arange(0,15*l0,0.1)
y_range = np.arange(-2*w0,2*w0,0.1)
extent1 = [x_range[0]/l0, x_range[-1]/l0,theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360]
extent2 = [r_range[0]/l0, r_range[-1]/l0, x_range[0]/l0, x_range[-1]/l0]
extent3 = [r_range[0]/l0, r_range[-1]/l0, theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360]

Y0, Z0 = np.meshgrid(y_range, y_range)
R0 = np.sqrt(Y0**2+Z0**2)
THETA0 = np.arctan2(Z0,Y0)
THETA1, X1 = np.meshgrid(theta_range, x_range)
R2, X2 = np.meshgrid(r_range, x_range)
R3, THETA3 = np.meshgrid(r_range, theta_range)

r1 = 1.5*l0
Ey, Ez, Ex = getE(r1,THETA1,X1,20*l0)
Y = r1*np.cos(THETA1)
Z = r1*np.sin(THETA1)
Troque = - (Y*Ez - Z*Ey)

plt.imshow(Troque,cmap="RdYlBu",aspect="auto",origin="lower")
plt.colorbar()


def Lorentz_net_gain(coord):
    dr, dtheta, dx = lambda r,theta,x,t:0,lambda r,theta,x,t:0,lambda r,theta,x,t:0

    dr = dr_func
    max_Lx = []
    min_Lx = []
    mean_Lx = []
    t_range = np.arange(0,40*l0,0.05)
    r_range = np.arange(0.*w0,1.75*w0,0.05)
    temp_env = gauss(t_range,x0)
    for r0 in tqdm(r_range):
        Lx_theta_list = []
        for theta0 in np.arange(0,2*pi,pi/64):
            Ey, Ez, Ex = getE(np.abs(r0+dr(r0,theta0,x0,t_range)),
                        theta0+dtheta(r0,theta0,x0,t_range),
                        x0+dx(r0,theta0,x0,t_range),
                        t_range)
            Y = r0*np.cos(theta0)
            Z = r0*np.sin(theta0)
            Troque = - (Y*Ez - Z*Ey)
            LxR_distrib = integrate.simpson(Troque*temp_env, x=t_range)
            Lx_theta_list.append(LxR_distrib)
        max_Lx.append(np.max(Lx_theta_list))
        min_Lx.append(np.min(Lx_theta_list))
        mean_Lx.append(np.mean(Lx_theta_list))
    return np.array(r_range),np.array(min_Lx),np.array(max_Lx), np.array(mean_Lx)

plt.figure()
coords_var = "r_theta"
d_r_list, min_Lx_model, max_Lx_model, mean_Lx_model = Lorentz_net_gain(coords_var)
plt.plot(d_r_list/l0,min_Lx_model,"C0--",lw=2)
plt.plot(d_r_list/l0,max_Lx_model,"C0--",lw=2, label=coords_var)
mean_max_modelS = np.mean([min_Lx_model,max_Lx_model],axis=0)

plt.plot(d_r_list/l0,mean_max_modelS,"C1")
plt.plot(d_r_list/l0,mean_Lx_model,"C2")
plt.legend()
plt.grid()
plt.xlabel("$r/\lambda$")
plt.ylabel("$L_x$")
plt.title("L_x(r) from Azimuthal electric field")
plt.xlim(0,2)

azeeazazeazeaze

theta_range = np.arange(0,2*pi,pi/64)
r_range = np.arange(0,2*w0,0.1)
x_range = np.arange(0,15*l0,0.1)


THETA1, X1 = np.meshgrid(theta_range, x_range)
R2, X2 = np.meshgrid(r_range, x_range)
R3, THETA3 = np.meshgrid(r_range, theta_range)

extent1 = [theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360, x_range[0]/l0, x_range[-1]/l0]
extent2 = [r_range[0]/l0, r_range[-1]/l0, x_range[0]/l0, x_range[-1]/l0]
extent3 = [r_range[0]/l0, r_range[-1]/l0, theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360]

# aezazeaez
plt.figure()
Ftheta_1 = Ftheta_V2_O5(1.5*l0, THETA1, X1)
plt.imshow(Ftheta_1,cmap="RdYlBu",aspect="auto",origin="lower", extent=extent1)
plt.colorbar()
plt.title("$F_\Theta$ (Theta & X)")
plt.xlabel("Theta")
plt.ylabel("X/$\lambda$")

plt.figure()
Ftheta_2 = Ftheta_V2_O5(R2, pi/4, X2)
plt.imshow(Ftheta_2,cmap="RdYlBu",aspect="auto",origin="lower", extent=extent2)
plt.colorbar()
plt.title("$F_\Theta$ (R & X)")
plt.xlabel("R/$\lambda$")
plt.ylabel("X/$\lambda$")

plt.figure()
Ftheta_3 = Ftheta_V2_O5(R3, THETA3, 5*l0)
plt.imshow(Ftheta_3,cmap="RdYlBu",aspect="auto",origin="lower", extent=extent3)
plt.colorbar()
plt.title("$F_\Theta$ (R & Theta)")
plt.xlabel("R/$\lambda$")
plt.ylabel("Theta")
# r_range_FM, Lx_FM = Lx_distrib_FullMotion_FV2O5()
# a_range, min_Lx_FM, max_Lx_FM = min_max(r_range_FM,Lx_FM)

# COEF_FM = sqrt(1+(a0*f(a_range,x0))**2+ 1/4*(a0*f(a_range,x0))**4)

# plt.plot(a_range/l0,COEF_FM*min_Lx_FM,"C1-",label="Lx FM ($\Delta \r, \Delta \Theta$, \Delta x$)")
# plt.plot(a_range/l0,COEF_FM*max_Lx_FM,"C1-")

# a_range, mean_Lx_FM, std_Lx_FM = averageAM(r_range_FM, Lx_FM,1)
# COEF_FM2 = sqrt(1+(a0*f(a_range,x0))**2+ 1/4*(a0*f(a_range,x0))**4)

# plt.plot(a_range/l0, COEF_FM2*mean_Lx_FM*10, "C3.-")

# plt.grid()
# plt.legend()

"""
r_range3, mean_Lx3, std_Lx3 = Lx4_distrib_mean_FULL_MOTION(N=50_000)
COEF3 = sqrt(1+(f(r_range3,x0)*a0)**2+ 1/4*(f(r_range3,x0)*a0)**4)
"""