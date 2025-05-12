# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 14:58:56 2025

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
from smilei_utils import dr_func, dx_func, theta_func, dtheta_func

plt.close("all")

l0 = 2*pi
print("--")
a0_requested = 3
sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a1_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a2.5_Tp12",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a3.5_Tp12_dx128_AM4",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4_Tp12_dx128_AM8",
                "SIM_OPTICAL_GAUSSIAN/gauss_a4.5_Tp12_dx128_AM4"]

a0_range_12 = np.array([0.1,1,2,2.5,3,3.5,4,4.5])

fig, axs = plt.subplots(2,len(sim_loc_list_12)//2, figsize=(10,5))
axs_list = axs.ravel()

for i,a0_requested in enumerate(a0_range_12):
    a0_sim_idx = np.where(a0_range_12==a0_requested)[0][0]
    sim_path = sim_loc_list_12[a0_sim_idx]
    ax = axs_list[i]
    
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
    
    RMIN, RMAX = 0.*l0,5.55*l0
    
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
        return 1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)
    
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
            temp_env = gauss(t_range, x[0, Nid]+dx_func(r[0, Nid], theta[0, Nid], x[0, Nid], t_range))
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
    
    
    def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
        return Tint*r*Ftheta_V2_O5(r,theta,z)
    
    def Lx4_distrib_GAMMA():
        t_range_lx = t_range#np.arange(0,t_range[-1],dt)
        temp_env = gauss(t_range_lx,x0)
        distrib_Lx_list = []
        for r in tqdm(grid_r):
            Lx_theta_list = []
            for theta in grid_theta:
                COEF_time = sqrt(1+(f(r+dr_func(r,t_range_lx,x0),x0)*a0)**2+ 1/4*(f(r+dr_func(r,t_range_lx,x0),x0)*a0)**4)
                LxR_distrib = integrate.simpson(COEF_time*Torque_V2_O5(np.abs(r+dr_func(r,t_range_lx,x0)), theta, x0)*temp_env**2, x=t_range_lx)
                Lx_theta_list.append(LxR_distrib)
            distrib_Lx_list.append(np.max(Lx_theta_list))
        return np.array(grid_r),np.array(distrib_Lx_list)
    
    # %% MAIN CODE
    
    PERCENTILE = 95
    
    dtheta_grid = 0.1
    dr_grid = 0.05
    grid_r = np.arange(0,2*l0,dr_grid)
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
    
    # Lx_max_model = np.max(LxEpolar_V2_O5(R,THETA,x0,w0,a0,3/8*Tp),axis=1)
    # COEF = sqrt(1+(a0*f(grid_r,x0))**2+ 1/4*(a0*f(grid_r,x0))**4)
    # ax.plot(grid_r/l0,COEF*Lx_max_model,"k-", lw=2)
    # ax.plot(grid_r/l0,-COEF*Lx_max_model,"k-", lw=2, label="Model $\gamma_{max}$$L_x^{NR}$")
    
    a_range, min_Smilei, max_Smilei = min_max_percentile(r[0],Lx_track[-1], dr_av=0.15,percentile=PERCENTILE)
    ax.plot(a_range/l0, min_Smilei, "C0",label="Smilei")
    ax.plot(a_range/l0, max_Smilei, "C0")
    ax.fill_between(a_range/l0, min_Smilei, max_Smilei, color="C0",alpha=0.25)
    
    grid_r, Lx_distrib = Lx4_distrib_GAMMA_MAX(grid_theta)
    a_range, min_Lx_R, max_Lx_R = min_max_percentile(grid_r,Lx_distrib, dr_av=0.15,percentile=PERCENTILE)
    ax.plot(a_range/l0,min_Lx_R,"k-", lw=2)
    ax.plot(a_range/l0,max_Lx_R,"k-", lw=2, label="Model $\gamma_{max}$$L_x^{R}$")
    
    # plt.scatter(r[0]/l0,Lx_track[-1],s=3,c="C0",label="Smilei")
    
    a_range, min_tot_Smilei, max_tot_Smilei = min_max_percentile(R,Lx_R_smilei+Lx_NT, dr_av=0.15,percentile=PERCENTILE)
    
    ax.plot(a_range/l0,max_tot_Smilei,"r-",lw=2,label="Numerical $L_x^{\perp,Smilei} + L_x^{||,Smilei}$")
    ax.plot(a_range/l0,min_tot_Smilei,"r-",lw=2)
    
    ax.plot(a_range/l0,max_Lx_tot_VTINT_dx,"C2",lw=2,label="Semi-numerical $L_x^{\perp,Smilei} + L_x^{||,R}$")
    ax.plot(a_range/l0,min_Lx_tot_VTINT_dx,"C2",lw=2)
    
    fs = 14

    ax.set_title(f"$\mathbf{{a_0={a0}}}$",fontsize=fs)
    
    # plt.plot(a_range/l0,max_Lx_tot_VTINT_TINT,"C1",lw=3,label="Semi-numerical $L_x^{\perp,Smilei} + L_x^{||,R}$ TINT")
    
    # plt.grid()
    # plt.legend()
    plt.xlabel("$r_0/\lambda$")
    plt.ylabel("$L_x$")
    
    # plt.title(f"{a0=}")
    # plt.xlim(0,2)
    # plt.ylim(bottom=-0.01)
    # plt.tight_layout()
    ax.set_xlim(0,2)

axs.ravel()[0].set_ylabel("$L_x$",fontsize=fs)
axs.ravel()[4].set_ylabel("$L_x$",fontsize=fs)

axs.ravel()[4].set_xlabel("$r_0/\lambda$",fontsize=fs)
axs.ravel()[5].set_xlabel("$r_0/\lambda$",fontsize=fs)
axs.ravel()[6].set_xlabel("$r_0/\lambda$",fontsize=fs)
axs.ravel()[7].set_xlabel("$r_0/\lambda$",fontsize=fs)



for i,ax in enumerate(axs.ravel()):
    handles, previous_labels = ax.get_legend_handles_labels()
    new_labels = ["Smilei", "$\gamma_{max} L_x^{R}$", "Numerical $L_x^{\perp,Smilei} + L_x^{||,Smilei}$","Semi-numerical $L_x^{\perp,Smilei} + L_x^{||,R}$"]
    ax.legend(loc="lower center",handles=handles, labels=new_labels,ncol=5, 
              columnspacing=0.6,handlelength=1.2,fontsize=11, 
              bbox_to_anchor=(2.3, -0.53), frameon=1)
    if i!=0: ax.get_legend().remove()
plt.subplots_adjust(hspace=0.7)  
plt.tight_layout()
