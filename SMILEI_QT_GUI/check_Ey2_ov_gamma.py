# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:24:25 2025

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
from smilei_utils import averageAM, min_max,min_max_percentile

plt.close("all")

l0=2*pi

a0_requested = 1
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


# sim_path = "sim_NEW_TERM_a4_dx128_AM4"
# sim_path = "sim_NEW_TERM_a4_dx64_AM4"
sim_path = "SIM_NEW_TERM/sim_NEW_TERM_a3_dx128_AM4_HD"

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
    
x = track_traj["x"][:,::N_part]
track_N = x.shape[1]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]
Ey = track_traj["Ey"]
Ex = track_traj["Ex"]

r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
# p_theta = p_theta = Lx_track/r
theta = np.arctan2(z,y)

x0 = x[0,0]


# Term = np.mean(Ey[1100:1250]**2,axis=0)
New_Term = integrate.simpson(Ey**2/gamma,x=t_range,axis=0)

plt.figure()
plt.scatter(y[0]/l0,z[0]/l0,c=New_Term,s=2,cmap="jet")
plt.xlabel("$y_0/\lambda$")
plt.ylabel("$z_0/\lambda$")
plt.title("$\int_0^{\infty}E_y^2/\gamma~dt$")
plt.colorbar()
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.tight_layout()

grid = np.arange(-w0,w0,0.05)
extent = [grid[0]/l0,grid[-1]/l0,grid[0]/l0,grid[-1]/l0]
grid_z2 = griddata((y[0],z[0]), New_Term, (grid[None, :], grid[:, None]))
plt.figure()
plt.imshow(grid_z2,aspect="auto",origin="lower",cmap="jet",extent=extent)
plt.colorbar()
plt.xlabel("$y_0/\lambda$")
plt.ylabel("$z_0/\lambda$")
plt.title("$\int_0^{\infty}E_y^2/\gamma~dt$")
plt.tight_layout()

# plt.figure()
# plt.scatter(theta[0], r[0]/l0,c=Term,cmap="jet",s=3)
# plt.colorbar()

grid_r = np.arange(0,w0,0.03)
grid_theta = np.arange(-pi,pi,0.2)

extent3 = [grid_theta[0],grid_theta[-1],grid_r[0]/l0,grid_r[-1]/l0]
grid_z3 = griddata((theta[0],r[0]), New_Term, (grid_theta[None, :], grid_r[:, None]))
plt.figure()
plt.imshow(grid_z3,aspect="auto",origin="lower",cmap="jet",extent=extent3)
plt.colorbar()
plt.xlabel("$\Theta_0$")
plt.ylabel("$r_0/\lambda$")
plt.title("$\int_0^{\infty}E_y^2/\gamma~dt$")
plt.tight_layout()

VMAX = 2
plt.figure()

Lx_NT = -1/4*np.gradient(grid_z3,grid_theta,axis=1)

# Ftheta = -1/4*np.gradient(grid_z3,grid_theta,axis=1)/grid_r[:,None]

plt.imshow(Lx_NT,aspect="auto",origin="lower",cmap="RdYlBu",
           extent=extent3, vmin=-VMAX, vmax=VMAX)
plt.colorbar()

plt.xlabel("$\Theta_0$")
plt.ylabel("$r_0/\lambda$")
plt.title("$L_x^{NT}$  from Transverse fields")
plt.tight_layout()




THETA, R  = np.meshgrid(grid_theta,grid_r)  # Create a grid


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

c = ax.pcolormesh(THETA, R, Lx_NT, shading='auto',cmap="RdYlBu", vmin=-VMAX, vmax=VMAX)

plt.colorbar(c, ax=ax, label='$L_x^{NT}$')

ax.set_title('Polar Plot $L_x^{NT}$  from Transverse fields')


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

def dr_Relat_mean(r,t,x):
    return -1/(1+(f(r,x)*a0)**2/2+ 1/4*1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr_Relat(r,t,x):
    return -1/(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)

def dr(r,t,x):
    return -1/sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)*a0**2/4*f_squared_prime(r, x)* gauss2_int_int(t,x)
def dx_func(r,theta,x,t):
    """ use 1/gamma ?"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return a0**2/4 * f(r,x)**2 * gauss2_int(t,x) 

def dtheta_func(r,theta,x,t):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return 1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)
    
def dr_func(r,theta,x,t):
    return dr_Relat_mean(r, t, x)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.pcolormesh(THETA, R, Torque_V2_O5(R, THETA, 5*l0)/R, shading='auto',cmap="RdYlBu", vmin=-VMAX, vmax=VMAX)
plt.colorbar(c, ax=ax, label='$F_\Theta$')
ax.set_title('Polar Plot $F_\Theta$  from Longitudinal fields')

dtheta = 0.1
grid_r = np.arange(0,w0,0.01)
grid_theta = np.arange(-pi,pi,dtheta)
THETA, R  = np.meshgrid(grid_theta,grid_r)  # Create a grid

def Lx4_distrib_GAMMA_MAX():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    distrib_Lx_list = np.zeros_like(Lx_NT)
    for i,r in enumerate(grid_r):
        for j,theta in enumerate(grid_theta):
            COEF_time = sqrt(1+(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**2+ 1/4*(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**4)

            LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(np.abs(r+dr(r,t_range_lx,x0)), theta, x0)*temp_env**2, x=t_range_lx)
            distrib_Lx_list[i,j] = LxR_distrib
    return np.array(grid_r),distrib_Lx_list

def Lx4_distrib_GAMMA_MEAN():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    distrib_Lx_list = np.zeros_like(Lx_NT)
    for i,r in enumerate(grid_r):
        for j,theta in enumerate(grid_theta):
            COEF_time = sqrt(1+(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**2/2+ 1/16*(f(r+dr_Relat_mean(r,t_range_lx,x0),x0)*a0)**4)

            LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(np.abs(r+dr(r,t_range_lx,x0)), theta, x0)*temp_env**2, x=t_range_lx)
            distrib_Lx_list[i,j] = LxR_distrib
    return np.array(grid_r),distrib_Lx_list

def Lx4_distrib_GAMMA_FULL():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x0)
    Lx_full = []
    for nid in tqdm(range(r.shape[-1])):
        temp_env = gauss(t_range,x[:,nid])
        LxR_distrib = integrate.simpson(Torque_V2_O5(r[:,nid], theta[:,nid], x[:,nid])*temp_env**2, x=t_range_lx)
        Lx_full.append(LxR_distrib)
    return np.array(Lx_full)

def Lx_distrib_FullMotion_FV2O5():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in tqdm(range(len(x[0]))):
        temp_env = gauss(t_range,x[0,Nid]+dx_func(r[0,Nid], theta[0,Nid],x[0,Nid],t_range))
        LxR_distrib = integrate.simpson(Torque_V2_O5(np.abs(r[0,Nid]+dr_Relat_mean(r[0,Nid],t_range_lx,x0)), theta[0,Nid], x0)*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)




Tint = 3/8*Tp
COEF = np.sqrt(1 + (f(R,x0)*a0)**2 + 1/4*(f(R,x0)*a0)**4)
COEF2 = np.sqrt(1 + (f(R,x0)*a0)**2/2 + 1/16*(f(R,x0)*a0)**4)
COEF3 = np.sqrt(1 + (f(R,x0)*a0)**2/2)

old_term = integrate.simpson(Ex**2,x=t_range,axis=0)
old_term_interp = griddata((theta[0],r[0]), old_term, (grid_theta[None, :], grid_r[:, None]))

New_Term = integrate.simpson(Ey**2/gamma,x=t_range,axis=0)
grid_z3 = griddata((theta[0],r[0]), New_Term, (grid_theta[None, :], grid_r[:, None]))



Lx_R_smilei = -1/2*np.gradient(old_term_interp,grid_theta,axis=1)
Num_Coef = COEF
Lx_NT = -1/4*np.gradient(grid_z3,grid_theta,axis=1)/Num_Coef**2


a_range, Lx_model_R_mean = Lx4_distrib_GAMMA_MEAN()
a_range, min_Lx_Rmean, max_Lx_Rmean = min_max(grid_r,Lx_model_R_mean, dr_av=0.22)

a_range, Lx_model_R_max = Lx4_distrib_GAMMA_MAX()
a_range, min_Lx_Rmax, max_Lx_Rmax = min_max(grid_r,Lx_model_R_max, dr_av=0.22)

d_r_list, d_Lx_list = Lx_distrib_FullMotion_FV2O5()
# a_range, lower_Lx, upper_Lx = min_max(d_r_list,d_Lx_list, dr_av=0.22)

total_Lx = Lx_model_R_mean + Lx_NT
a_range, min_Lx_tot, max_Lx_tot = min_max_percentile(grid_r,total_Lx, dr_av=0.22,percentile=95)

Lx_Tint_interp = griddata((theta[0],r[0]), d_Lx_list, (grid_theta[None, :], grid_r[:, None]))
total_Lx_VTINT = Lx_Tint_interp + Lx_NT
a_range, min_Lx_tot_VTINT, max_Lx_tot_VTINT = min_max_percentile(R,total_Lx_VTINT, dr_av=0.22,percentile=95)

plt.figure()

plt.scatter(r[0]/l0,Lx_track[-1],s=3,c="C0",label="Smilei")

a_range, min_tot_Smilei, max_tot_Smilei = min_max_percentile(R,Lx_R_smilei+Lx_NT, dr_av=0.05,percentile=93)

plt.plot(a_range/l0,max_tot_Smilei,"r--",lw=2,label="$L_x^{Smilei} + L_x^{NT}$")
plt.plot(a_range/l0,min_tot_Smilei,"r--",lw=2)

plt.plot(a_range/l0,max_Lx_Rmax,"k",lw=3,label="$L_x^{R}$ with $\gamma_{max}$")
plt.plot(a_range/l0,min_Lx_Rmax,"k",lw=3)

plt.plot(a_range/l0,max_Lx_tot,"C1",lw=3,label="$L_x^{R} + L_x^{NT}$ with $<\gamma>$")
plt.plot(a_range/l0,min_Lx_tot,"C1",lw=3)

plt.plot(a_range/l0,max_Lx_tot_VTINT,"C2",lw=3,label="Using accurate \nmodel for $Lx_R$ (Tint)")
plt.plot(a_range/l0,min_Lx_tot_VTINT,"C2",lw=3)

plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")

plt.title(f"{a0=}, N={Lx_track.shape[-1]}, $\Delta \Theta_{{interp}}$={(grid_theta[1]-grid_theta[0]):.2f}, max num coef = {np.max(Num_Coef):.1f}")
plt.xlim(0,2.5)
plt.ylim(bottom=-0.01)
plt.tight_layout()


