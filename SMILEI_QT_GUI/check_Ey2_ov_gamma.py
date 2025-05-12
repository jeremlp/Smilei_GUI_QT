# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:24:25 2025

@author: Jeremy
"""

# %% INIT

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
from smilei_utils import SET_CONSTANTS, averageAM, min_max,min_max_percentile
#from smilei_utils import dr_func, dx_func, theta_func, dtheta_func

plt.close("all")

l0 = 2*pi
print("--")
a0_requested = 3
sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.33_Tp12_dx48",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx128_AM4",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx128_AM4"]

a0_range_12 = np.array([0.1,1,2,2.33,2.5,3,3.5,4,4.5])

a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
sim_path = sim_loc_list_12[a0_sim_idx]


# sim_path = "sim_NEW_TERM_a4_dx64_AM4"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a2_dx128_AM4_HD"

sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_HD"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_UHD"
# sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a2.5_dx128_AM4_HD"

sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_l-1"
sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_l2"

# sim_path = "SIM_NET_GAIN/gauss_a3_Tp12_NET_GAIN_dx128_AM4_L-1"


N_part = 1
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ey","Ex"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = 1
track_N_tot = T0.nParticles
t_range = T0.getTimes()
track_traj = T0.getData()
if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2

y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
r = np.sqrt(y**2 + z**2)

RMIN, RMAX = 0.*l0,3*l0

mask = np.where((r[0]>RMIN) & (r[0] < RMAX) )[0]
TMAX = None

x = track_traj["x"][:TMAX,mask]
y = track_traj["y"][:TMAX,mask]-track_r_center
z = track_traj["z"][:TMAX,mask] -track_r_center
track_N = x.shape[1]

py = track_traj["py"][:TMAX,mask]
pz = track_traj["pz"][:TMAX,mask]
px = track_traj["px"][:TMAX,mask]
Ey = track_traj["Ey"][:TMAX,mask]
Ex = track_traj["Ex"][:TMAX,mask]

r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
# p_theta = p_theta = Lx_track/r
theta = np.arctan2(z,y)

x0 = x[0,0]

SET_CONSTANTS(w0, a0, Tp)

# %% FUNCTIONS

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

def Ftheta_V2_O3(r,theta,z):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3
# @njit
def Ftheta_V2_O5(r,theta,z):
    numerator = (
        2 * a0**2 * r * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
        (2 * z * np.cos(2 * theta) * ( 4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)) +
            np.sin(2 * theta) * (  4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                r**2 * ( 8 * (w0**4 + 4 * w0**2 + 20) * z**2 + (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4) -
                (w0**4 + 4 * z**2) * ( 4 * (w0**2 - 2) * z**2 + (w0**2 + 4) * (w0**2 + 6) * w0**2  ))))
    denominator = (w0**4 + 4 * z**2)**5
    return numerator / denominator


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
g = g_gauss
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
def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)

def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x0)/sigma_gauss)+1)
def gauss2_int_int(t,x):
    t_center = 1.25*Tp
    psi = lambda t,x : sqrt(2)*(t-t_center-x)/sigma_gauss
    expr =lambda t,x : 0.5*sqrt(pi/2)*sigma_gauss*( gauss(t,x)**2*sigma_gauss/sqrt(2*pi) + t + (t-t_center-x)*(special.erf(psi(t,x))))
    return expr(t,x) - expr(0,x)

def gauss2_int(t,x):
    return 0.5*sqrt(pi/2)*sigma_gauss* (special.erf(sqrt(2)*(t-t_center-x0)/sigma_gauss)+1)
def gauss2_int_int(t,x):
    t_center = 1.25*Tp
    psi = lambda t,x : sqrt(2)*(t-t_center-x)/sigma_gauss
    expr =lambda t,x : 0.5*sqrt(pi/2)*sigma_gauss*( gauss(t,x)**2*sigma_gauss/sqrt(2*pi) + t + (t-t_center-x)*(special.erf(psi(t,x))))
    return expr(t,x) - expr(0,x)


def dx_func(r,theta,x,t):
    """ use 1/gamma ?"""
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return 1/gamma * a0**2/4 * f(r,x)**2 * gauss2_int(t,x) 

def dtheta_func(r,theta,x,t):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)

def theta_func(r,theta,x,t):
    r_model = np.abs(r + dr_func(r,x,t))
    x_model = x0 + dx_func(r, theta, x, t)

    # idx_cross = np.where(r_model==np.min(r_model))[0][0]
    theta0_m = np.zeros_like(r_model) + theta
    # theta0_m[idx_cross:] = pi+theta0_m[idx_cross:] #Allow model to cross beam axis and switch theta

    y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(t,x_model)*cos(l1*theta0_m  - t + x_model)
    z_model = r_model*sin(theta0_m)

    theta_model = np.arctan2(z_model, y_model) + dtheta_func(r, theta0_m, x_model, t)

    return theta_model
    
def dr_func_small_r(r,theta,x,t):
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma**2 *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func_large_r(r,theta,x,t):
    """For large r, using 1/gamma works better than the true 1/gamma^2 """
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1/gamma *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func_test_r(r,theta,x,t):
    """For test r, using 1 """
    gamma = sqrt(1+(f(r,x)*a0)**2/2+ 1/16*(f(r,x)*a0)**4)
    return -1 *a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_func(r,x,t):
    return dr_func_large_r(r,theta,x,t)


# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# c = ax.pcolormesh(THETA, R, Torque_V2_O5(R, THETA, 5*l0)/R, shading='auto',cmap="RdYlBu", vmin=-VMAX, vmax=VMAX)
# plt.colorbar(c, ax=ax, label='$F_\Theta$')
# ax.set_title('Polar Plot $F_\Theta$  from Longitudinal fields')


def Lx4_distrib_GAMMA_MAX(grid_theta):
    t_range_lx = t_range  # np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx, x0)
    distrib_Lx_list = np.zeros_like(Lx_NT)
    for i, r in tqdm(enumerate(grid_r)):
        for j, theta in enumerate(grid_theta):
            COEF_time = sqrt(1+(f(r+dr_func(r, x0, t_range_lx), x0)*a0)
                             ** 2 + 1/4*(f(r+dr_func(r, x0, t_range_lx), x0)*a0)**4)

            LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(
                np.abs(r+dr_func(r, x0, t_range_lx)), theta, x0)*temp_env**2, x=t_range_lx)
            distrib_Lx_list[i, j] = LxR_distrib
    return np.array(grid_r), distrib_Lx_list


def Lx_distrib_FV2O5(a0):
    t_range_lx = t_range  # np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        temp_env = gauss(t_range, x[0, Nid])
        LxR_distrib = integrate.simpson(Torque_V2_O5(
            r[0, Nid], theta[0, Nid], x0)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0, Nid])
    return np.array(distrib_r_list), np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_FV2O5_TINT(a0):
    t_range_lx = t_range  # np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        temp_env = gauss(t_range, x[:, Nid])
        LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(
            r[0, Nid]+dr_func(r[0, Nid], x0, t_range_lx)), theta[0, Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0, Nid])
    return np.array(distrib_r_list), np.array(distrib_Lx_list)

def Lx_distrib_FullMotion_FV2O5_dx():
    t_range_lx = t_range  # np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        temp_env = gauss(
            t_range, x[0, Nid]+dx_func(r[0, Nid], theta[0, Nid], x[0, Nid], t_range))
        LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(r[0, Nid]+dr_func(r[0, Nid], x0, t_range_lx)), theta[0, Nid], x[0, Nid]+dx_func(
            r[0, Nid], theta[0, Nid], x0, t_range_lx))*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0, Nid])
    return np.array(distrib_r_list), np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_FV2O5_dtheta():
    t_range_lx = t_range  # np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        temp_env = gauss(
            t_range, x[0, Nid]+dx_func(r[0, Nid], theta[0, Nid], x[0, Nid], t_range))
        LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(r[0, Nid]+dr_func(r[0, Nid], x0, t_range_lx)), theta_func(
            r[0, Nid], theta[0, Nid], x0, t_range), x[0, Nid]+dx_func(r[0, Nid], theta[0, Nid], x0, t_range_lx))*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0, Nid])
    return np.array(distrib_r_list), np.array(distrib_Lx_list)

# %% MAIN CODE

PERCENTILE = 95

dtheta_grid = 0.1
dr_grid = 0.05
grid_r = np.arange(RMIN,RMAX,dr_grid)
grid_theta = np.arange(-pi,pi,dtheta_grid)
THETA, R  = np.meshgrid(grid_theta,grid_r)  # Create a grid

Tint = 3/8*Tp
COEF_MAX = np.sqrt(1 + (f(R,x0)*a0)**2 + 1/4*(f(R,x0)*a0)**4)
COEF_MEAN = np.sqrt(1 + (f(R,x0)*a0)**2/2 + 1/16*(f(R,x0)*a0)**4)

#--------------------------------------
# LONGITUDINAL TERM SMILEI INTEGRATION
#--------------------------------------
old_term = integrate.simpson(Ex**2,x=t_range,axis=0)
old_term_interp = griddata((theta[0],r[0]), old_term, (grid_theta[None, :], grid_r[:, None]))
Lx_R_smilei = -1/2*np.gradient(old_term_interp,grid_theta,axis=1)
#--------------------------------------
# NEW TERM TRANSVERSE SMILEI INTEGRATION
#--------------------------------------
New_Term = integrate.simpson(Ey**2,x=t_range,axis=0)
grid_z3 = griddata((theta[0],r[0]), New_Term, (grid_theta[None, :], grid_r[:, None]))
Num_Coef = pi
Lx_NT = -1/2*np.gradient(grid_z3,grid_theta,axis=1)/COEF_MEAN/Num_Coef

d_r_list_dx, d_Lx_list_dx = Lx_distrib_FullMotion_FV2O5_dx()
Lx_Tint_interp_dx = griddata((theta[0],r[0]), d_Lx_list_dx, (grid_theta[None, :], grid_r[:, None]))
a_range, min_Lx_tot_VTINT_dx, max_Lx_tot_VTINT_dx = min_max_percentile(R,Lx_Tint_interp_dx + Lx_NT, dr_av=0.22,percentile=PERCENTILE)

d_r_list_TINT, d_Lx_list_TINT = Lx_distrib_FullMotion_FV2O5_TINT(a0)
Lx_Tint_interp_TINT = griddata((theta[0],r[0]), d_Lx_list_TINT, (grid_theta[None, :], grid_r[:, None]))
a_range, min_Lx_tot_VTINT_TINT, max_Lx_tot_VTINT_TINT = min_max_percentile(R,Lx_Tint_interp_TINT + Lx_NT, dr_av=0.22,percentile=PERCENTILE)


plt.figure()

# plt.scatter(r[0]/l0,Lx_track[-1],s=3,c="C0",label="Smilei")
a_range, min_Smilei, max_Smilei = min_max_percentile(r[0],Lx_track[-1], dr_av=0.15,percentile=PERCENTILE)
plt.plot(a_range/l0, min_Smilei, "C0",label="Smilei")
plt.plot(a_range/l0, max_Smilei, "C0")
plt.fill_between(a_range/l0, min_Smilei, max_Smilei, color="C0",alpha=0.25)

a_range, min_tot_Smilei, max_tot_Smilei = min_max_percentile(R,Lx_R_smilei+Lx_NT, dr_av=0.15,percentile=PERCENTILE)

plt.plot(a_range/l0,max_tot_Smilei,"r-",lw=2,label="Numerical $L_x^{\perp,Smilei} + L_x^{||,Smilei}$")

plt.plot(a_range/l0,max_Lx_tot_VTINT_dx,"C2",lw=3,label="Semi-numerical $L_x^{\perp,Smilei} + L_x^{||,R}$ dx")

plt.plot(a_range/l0,max_Lx_tot_VTINT_TINT,"C1",lw=3,label="Semi-numerical $L_x^{\perp,Smilei} + L_x^{||,R}$ TINT")


plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")

plt.title(f"{a0=}, N={Lx_track.shape[-1]}, $\Delta \Theta_{{interp}}$={(grid_theta[1]-grid_theta[0]):.2f}, percentile={PERCENTILE}\n max num coef = {np.max(Num_Coef):.2f}")
plt.xlim(0,2)
plt.ylim(bottom=-0.01)
plt.tight_layout()
# plt.xlim(1,2)



a_range, min_Lx_R_Smilei, max_Lx_R_Smilei = min_max_percentile(R,Lx_R_smilei, dr_av=0.5,percentile=PERCENTILE)
a_range, min_Lx_NT, max_Lx_NT = min_max_percentile(R,Lx_NT, dr_av=0.22,percentile=PERCENTILE)
plt.figure()
plt.plot(a_range/l0,-min_Lx_NT)
plt.plot(a_range/l0,max_Lx_NT)

plt.plot(a_range/l0,-min_Lx_R_Smilei,"--")
plt.plot(a_range/l0,max_Lx_R_Smilei,"--")

plt.plot(a_range/l0,max_Lx_NT/2+max_Lx_R_Smilei,"k")
plt.grid()


"""
#===========================================
# EXACT DEFINITION
#===========================================
# dtheta_grid = 0.2
# dr_grid = 0.1
# grid_r = np.arange(0,2*l0,dr_grid)
# grid_theta = np.arange(-pi,pi,dtheta_grid)
# THETA, R  = np.meshgrid(grid_theta,grid_r)  # Create a grid

inside_term_time = []
for t in tqdm(range(0,len(t_range))):
    # print(t/len(t_range)*100)
    grid_Ey2 = griddata((theta[0],r[0]), Ey[t]**2, (grid_theta[None, :], grid_r[:, None]))
    grid_gamma = griddata((theta[0],r[0]), 1/gamma[t], (grid_theta[None, :], grid_r[:, None]))
    inside_term = grid_gamma * np.gradient(grid_Ey2,grid_theta,axis=1)
    inside_term_time.append(inside_term)

Lx_NT_exact = integrate.simpson(np.array(inside_term_time),x=t_range,axis=0)/pi

a_range, _, max_LxNT_Exact = min_max_percentile(R,Lx_R_smilei+Lx_NT_exact, dr_av=0.05,percentile=PERCENTILE)

plt.plot(a_range/l0,max_LxNT_Exact,"rx",lw=2,label="$L_x^{Smilei} + L_x^{NT}$ Exact")



"""
# %% NET GAIN
plt.figure()
a_range_av, av, _ = averageAM(r[0],Lx_track[-1],0.5)
plt.plot(a_range_av/l0, av,label="$<L_x>$ Smilei")

# a_range_av, av, _ = averageAM(R, Lx_NT*2,0.3)
# plt.plot(a_range_av/l0, -av,"--",label="$-<2 ~L_x^{\perp}>$")

# a_range_av, av, _ = averageAM(R, Lx_R_smilei,0.3)
# plt.plot(a_range_av/l0, -av,"--",label="$-<L_x^{\parallel}>$")
a_range_av, av, _ = averageAM(R, Lx_Tint_interp_dx+Lx_NT*2,0.3)
plt.plot(a_range_av/l0, -av,"--", label="$-<L_x^{\parallel} + 2 ~L_x^{\perp}>$ semi-num (dx)")

a_range_av, av, _ = averageAM(R, Lx_R_smilei+Lx_NT*2,0.3)
plt.plot(a_range_av/l0, -av,label="$-<L_x^{\parallel} + 2 ~L_x^{\perp}>$ num")

plt.xlabel("$r_0/\lambda$")
plt.ylabel("$<L_x>$")
plt.title(f"Net gain $<L_x>$ ($a_0$={a0})")
plt.grid()
plt.legend()
plt.tight_layout()

plt.figure()
a_range_av, av, _ = averageAM(r[0],Lx_track[-1],0.5)
plt.plot(a_range_av/l0, av,label="$<L_x>$ Smilei")

a_range_av, av, _ = averageAM(R, Lx_NT*2,0.3)
plt.plot(a_range_av/l0, -av,"--",label="$-<2 ~L_x^{\perp}>$")

a_range_av, av, _ = averageAM(R, Lx_Tint_interp_dx,0.3)
plt.plot(a_range_av/l0, -av,"--",label="$-<L_x^{\parallel}> model$")

a_range_av, av, _ = averageAM(R, Lx_R_smilei,0.3)
plt.plot(a_range_av/l0, -av,"--",label="$-<L_x^{\parallel}>$ num")

# a_range_av, av, _ = averageAM(R, Lx_R_smilei+Lx_NT*2,0.3)
# plt.plot(a_range_av/l0, -av,label="$-<L_x^{\parallel} + 2 ~L_x^{\perp}>$ num")

plt.xlabel("$r_0/\lambda$")
plt.ylabel("$<L_x>$")
plt.title(f"Net gain $<L_x>$ ($a_0$={a0})")
plt.grid()
plt.legend()
plt.tight_layout()









# %% OTHER
"""
eazaezaezaze
#===========================================
# FOR MODEL OF NEW TERM
#===========================================


THETA2,R2,TIME2 = np.meshgrid(grid_theta, grid_r,t_range,indexing="ij")

x_model = x0 + dx_func(R2, THETA2, x0, TIME2)
r_model = np.abs(R2 + dr_func(R2,x0,TIME2))
theta0_m = np.zeros_like(r_model) + THETA2
#========================================================

# COEF3


# Lx_NT_FM = Lx_distrib_FullMotion_NT(Num_Coef, Pondero_Factor)
a_range, _, max_LxNT_FM = min_max_percentile(R,Lx_NT_FM, dr_av=0.05,percentile=PERCENTILE)

y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x0)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model)

Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)/COEF2.T
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF_X0 = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x_model)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x_model, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)/COEF2.T

a_range, _, max_Lx_tot_model_Rmodel_arctan_pondero_LF_Xmodel = min_max_percentile(R,Lx_Tint_interp + Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

# Lx_NT_only = -1/2*np.gradient(grid_z3,grid_theta,axis=1)
# a_range, _, max_LxNT_only_smilei = min_max_percentile(R,Lx_NT_only, dr_av=0.05,percentile=93)




plt.figure()
plt.scatter(r[0]/l0,Lx_track[-1],s=3,c="C0",label="Smilei")
plt.plot(a_range/l0,max_Lx_Rmax,"k",lw=3,label="Old model $L_x^{R}$ with $\gamma_{max}$")

plt.plot(a_range/l0,max_tot_Smilei,"r--",lw=2,label="Numerical $L_x^{Smilei} + L_x^{NT}$")
plt.plot(a_range/l0,max_Lx_tot_VTINT,"C1",lw=3,label="Semi-Model $L_x^{R} + L_x^{NT}$")
# plt.plot(a_range/l0,max_Lx_tot_model_Rmodel_arctan_pondero_LF_Xmodel,"C2-", label="Model $L_x^{R} + L_x^{NT,R}$")

# a_range, _, max_Lx_tot_model_Rmodel_arctan_pondero_LF_Xmodel = min_max_percentile(R,Lx_Tint_interp + Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)
# plt.plot(a_range/l0,max_Lx_tot_model_Rmodel_arctan_pondero_LF_Xmodel,"C2-",lw=2, label="Model $L_x^{R} + L_x^{NT,R}$")
# plt.plot(a_range/l0,max_LxNT_FM,"r--",label="$L_x^{NT}$ FM")

N_density = r[0].shape[0]/(np.max(r[0])-np.min(r[0]))
plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.ylim(bottom=0)
plt.title(f"{a0=}, $N/\Delta R$={N_density:.0f}, $\Delta \Theta_{{interp}}$={dtheta_grid:.2f}, percentile={PERCENTILE}\n max num coef = {np.max(Num_Coef):.2f}, pondero factor = {Pondero_Factor:.2f}")
plt.xlim(0,2.1)
plt.tight_layout()









aaezazeaez





a_range, min_LxNT_smilei, max_LxNT_smilei = min_max_percentile(R,Lx_NT, dr_av=0.05,percentile=PERCENTILE)

y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x0)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model)

Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF_X0 = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x_model)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x_model, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

plt.figure()
plt.plot(a_range/l0,max_LxNT_smilei, label="Smilei NT")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_pondero_LF_X0,"-", label="Model NT r_model + arctan + pondero + LF + x_model")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel,"-", label="Model NT r_model + arctan + pondero + LF + x_model")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.ylim(bottom=-0.01)
plt.title("NT from Smilei numerical integration VS model") 
plt.tight_layout()

















THETA2,R2,TIME2 = np.meshgrid(grid_theta, grid_r,t_range,indexing="ij")

x_model = x0 + dx_func(R2, THETA2, x0, TIME2)
r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
theta0_m = np.zeros_like(r_model) + THETA2

Ey_M,Ez_M,Ex_M = getE(R2, THETA2, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Nothing = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
Ey_M,Ez_M,Ex_M = getE(r_model, THETA2, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)


r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
y_model = r_model*cos(theta0_m)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)


r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x0)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model) 
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_LF = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(TIME2,x0)*cos(l1*theta0_m  - TIME2 + x0)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model) + dtheta_func(r_model, theta0_m, x0, TIME2)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)

r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x0)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model) + dtheta_func(r_model, theta0_m, x0, TIME2)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)


r_model = np.abs(R2 + dr_func(R2,THETA2,x0,TIME2))
y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(TIME2,x_model)*cos(l1*theta0_m  - TIME2 + x_model)
z_model = r_model*sin(theta0_m)
theta_model = np.arctan2(z_model, y_model) + dtheta_func(r_model, theta0_m, x0, TIME2)
Ey_M,Ez_M,Ex_M = getE(r_model, theta_model, x0, TIME2)
New_Term_Model = integrate.simpson(Ey_M**2,axis=-1)
Lx_NT_M = -1/2*np.gradient(New_Term_Model, grid_theta, axis=0)
a_range, _, max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel_V2 = min_max_percentile(R,Lx_NT_M.T, dr_av=0.05,percentile=PERCENTILE)



plt.figure()
plt.plot(a_range/l0,max_LxNT_smilei, label="Smilei NT")
plt.plot(a_range/l0,max_LxNT_model_Nothing, label="Model NT Nothing")
plt.plot(a_range/l0,max_LxNT_model_Rmodel, label="Model NT r_model")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan,"--", label="Model NT r_model + arctan")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_LF,"--", label="Model NT r_model + arctan + LF")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_pondero_LF,"-", label="Model NT r_model + arctan + pondero + LF")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel,"-", label="Model NT r_model + arctan + pondero + LF + x_model")
plt.plot(a_range/l0,max_LxNT_model_Rmodel_arctan_pondero_LF_Xmodel_V2,"-", label="Model NT r_model + arctan + pondero + LF + x_model")

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
plt.ylim(bottom=-0.01)
plt.title("NT from Smilei numerical integration VS model") 
plt.tight_layout()
"""