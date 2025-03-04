# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:32:11 2025

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

l0=2*pi

a0_requested = 2
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

N_part = 1
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])


Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
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
r = np.sqrt(y**2 + z**2)
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
p_theta = p_theta = Lx_track/r
theta = np.arctan2(z,y)

x0 = x[0,0]


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
t_center=1.25*Tp
sigma_gauss = Tp*3/8/c_gauss

def sin2(t,x):
    return sin(pi*(t-x)/Tp)**2*((t-x)<Tp)*((t-x)>0)
def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)
def getE(r,theta,z,t):
    tau = t-z
    g_ = gauss(t,z)

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
    return a0**2/4 * f(r,x)**2 * gauss2_int(t,x) /gamma

def dtheta_func(r,theta,x,t):
    """ possible 1/r missing for theta velocity"""
    gamma = sqrt(1+(f(r,x)*a0)**2+ 1/4*(f(r,x)*a0)**4)
    return 1/gamma * 1/r * Ftheta_V2_O5(r,theta,x) * gauss2_int_int(t, x)
    
def dr_func(r,theta,x,t):
    return dr_Relat_mean(r, t, x)

r0_requested = 1.55*l0
Nid = np.where(np.abs(r[0]-r0_requested) == np.min(np.abs(r[0]-r0_requested)))[0][0]
r0 = r[0,Nid]
theta0 = theta[0,Nid]
z0 = z[0,Nid]
y0 = y[0,Nid]

r_model = np.abs(r0+dr_func(r0, theta0, x0, t_range))
plt.figure()
plt.plot(t_range/l0, r[:,Nid],label="Smilei r")
plt.plot(t_range/l0, r_model,"--",label="Model r")
plt.grid()
plt.legend()


plt.figure()
plt.plot(t_range/l0, x[:,Nid],label="Smilei x")
plt.plot(t_range/l0, x0+dx_func(r0, theta0, x0, t_range),"--",label="Model r")
plt.grid()
plt.legend()


plt.figure()
plt.plot(t_range/l0, theta[:,Nid],label="Smilei $\Theta$")
plt.plot(t_range/l0, theta0+dtheta_func(r0, theta0, x0, t_range),"--",label="Model $\Theta$")
plt.plot(t_range/l0, theta0+dtheta_func(r0+r_model, theta0, x0, t_range),"--",label="Model $\Theta$ (with dr)")
plt.plot(t_range/l0, theta0+dtheta_func(r0+r_model, theta0, x0+dx_func(r0, theta0, x0, t_range), t_range),"--",label="Model $\Theta$ (with dr)")
plt.plot(t_range/l0,theta0+0.2*np.cos(t_range-x0)*gauss(t_range, x0))
plt.grid()
plt.legend()

r_model = np.abs(r0 + dr_func(r0,theta0,x0, t_range))
x_model = x0 +dx_func(r0, theta0, x0, t_range)

idx_cross = np.where(r_model==np.min(r_model))[0][0]
theta0_m = np.zeros_like(r_model) + theta0
theta0_m[idx_cross:] = pi+theta0_m[idx_cross:] #Allow model to cross beam axis and switch theta

y_model = r_model*cos(theta0_m) + a0*f(r_model,x0)*gauss(t_range,x_model)*cos(l1*theta0_m  - t_range + x_model)
z_model = r_model*sin(theta0_m)

plt.close("all")

plt.figure()
plt.plot(t_range/l0, y[:,Nid],label="Smilei")
plt.plot(t_range/l0,y_model,label="Model")
plt.title(f"Y model {a0=}")
plt.grid()
plt.legend()
plt.xlabel("$t/(c\lambda)$")
plt.ylabel("y")
plt.tight_layout()

plt.figure()
plt.plot(t_range/l0, z[:,Nid],label="Smilei")
plt.plot(t_range/l0,z_model,label="Model")
plt.title(f"Z model {a0=}")
plt.grid()
plt.legend()
plt.xlabel("$t/(c\lambda)$")
plt.ylabel("z")
plt.tight_layout()



plt.figure()
theta_model = np.arctan2(z_model, y_model) + dtheta_func(r0, theta0_m, x_model, t_range)

plt.plot(t_range/l0,theta[:,Nid],label="Smilei")
plt.plot(t_range/l0, theta_model,label="Model")
plt.plot(t_range/l0, np.arctan2(z_model, y_model) + dtheta_func(r_model, np.arctan2(z_model, y_model), x_model, t_range),label="theta_model included in Ftheta")

plt.grid()
plt.title(f"Theta model {a0=}")
plt.xlabel("$t/(c\lambda)$")
plt.ylabel("$\Theta$")

# plt.ylim(-0.005,0.012)
plt.legend()
plt.tight_layout()



plt.close("all")

t_range = np.arange(10*l0,35*l0,0.5)
y_grid = np.arange(-w0,w0,0.2)
Y,Z,TIME = np.meshgrid(y_grid, y_grid,t_range,indexing="ij")
R = np.sqrt(Y**2+Z**2)
THETA = np.arctan2(Z,Y)


plt.figure()

plt.scatter(0,0,s=200,marker="x",c="r",label="beam axis")
plt.plot(y[:,Nid]/l0,z[:,Nid]/l0)
plt.scatter(y[0,Nid]/l0,z[0,Nid]/l0,s=100,marker="x",c="k",label="Initial position")

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.grid()
plt.xlabel("$y_0/\lambda$")
plt.ylabel("$z_0/\lambda$")
plt.title(f"Particle trajectory at $r_0={r0/l0:.2f}\lambda$ for $a_0=${a0}")
plt.legend()
plt.tight_layout()


t_range = np.arange(10*l0,35*l0,0.1)
r_grid = np.arange(0,w0,0.1)
theta_grid = np.arange(-pi,pi,0.2)
THETA,R,TIME = np.meshgrid(theta_grid, r_grid,t_range,indexing="ij")

plt.figure()

r_model = np.abs(R + dr_func(R,THETA,x0,TIME))
x_model = x0 + dx_func(R, THETA, x0, TIME)

# idx_cross = np.where(r_model==np.min(r_model))[0][0]
theta0_m = np.zeros_like(r_model) + THETA
# theta0_m[idx_cross:] = pi+theta0_m[idx_cross:] #Allow model to cross beam axis and switch theta

y_model = r_model*cos(theta0_m) + a0*f(r_model,x_model)*gauss(TIME,x_model)*cos(l1*theta0_m  - TIME + x_model)
z_model = r_model*sin(theta0_m)

theta_model = np.arctan2(z_model, y_model) + dtheta_func(r0, theta0_m, x_model, TIME)

Ey,Ez,Ex = getE(r_model, theta_model, x_model, TIME)

extent = [y_grid[0]/l0,y_grid[-1]/l0, y_grid[0]/l0,y_grid[-1]/l0]
plt.imshow(np.mean(Ey**2,axis=-1),aspect="auto",origin="lower",extent=extent,cmap="jet")
plt.colorbar()
circ = plt.Circle((0,0),radius=w0/sqrt(2)/l0,color="k",fill=False)
plt.gca().add_patch(circ)

plt.figure()

extent = [theta_grid[0],theta_grid[-1], r_grid[0]/l0,r_grid[-1]/l0]

New_Term_Model = integrate.simpson(Ey**2,axis=-1)

plt.imshow(New_Term_Model.T,aspect="auto",origin="lower",extent=extent,cmap="tab20b")
plt.colorbar()
plt.title("$\int_0^\infty E_y^2/\gamma~dt$ using $\Theta$ model")

Lx = -1/4*np.gradient(New_Term_Model, theta_grid, axis=0)

plt.figure()
plt.scatter(r[0]/l0, Lx_track[-1],s=1)
plt.scatter(R[:,:,0]/l0, Lx,s=1)
plt.grid()
plt.xlabel("$r_0/\lambda")
plt.ylabel("$L_x$")
plt.title("$L_x^{NT}$ model using $\Theta$ model")
