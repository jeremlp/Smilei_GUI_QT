
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:38:04 2024

@author: jerem
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from numba import njit

from numpy import abs, cos, sin, pi, exp, sqrt, arctan2
from numpy.linalg import norm
from math import factorial
import os,sys
from tqdm import tqdm
from scipy import integrate
import scipy
# plt.rcParams["figure.figsize"] = (10,10)
plt.close("all")
#==========================
# CONSTANTS
#==========================
q = -1
me = 1
rs = q/me
l0 = 2*pi

me = 9.1093837*10**-31
e = 1.60217663*10**-19
c = 299792458
eps0 = 8.854*10**-12
mu0 = 1.2566*10**-6

l0_SI   = 1e-6#0.8e-6                    # laser wavelength (SI)
wr   = 2.*np.pi * c/l0_SI     # reference angular frequency (SI)
K = me*c**2
N = eps0*me*wr**2/e**2
L = c/wr
KNL3 = K*N*L**3

c_super=sqrt(2)*scipy.special.gamma(11/10)
c_gauss = sqrt(pi/2)
c_square = 2
c_sin = 1#4/pi
@njit
def g_cst(tau):
    return 1.0
@njit
def g_superGauss(tau):
    return  exp( -(tau-Tp/2)**10/(Tp/sqrt(2.)/2/c_super)**10 ) 
@njit
def g_gauss(tau):
    return  exp( -(tau-Tp/2)**2/(Tp/sqrt(2.)/2/c_gauss)**2 ) 
@njit
def g_gaussMath(tau):
    return  exp(-tau**2/Tp**2)
@njit
def g_square(tau):
    return  (tau-Tp/4<Tp/c_square) * (tau-Tp/4>0)
@njit
def g_sin(tau):
    t_shift = (Tp-Tp/c_sin)/2
    return sin(pi*(tau-t_shift)/(Tp/c_sin)) * (tau-t_shift<Tp/c_sin) * (tau-t_shift>0)
@njit
def g_sin2(tau):
    return sin(pi*tau/Tp)**2 * (tau<Tp) * (tau>0)
@njit(parallel=True)
def get_EB_fast(x,y,z,t):
    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    tau = t-z
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l)+1)*arctan2(z,zR)
    a_ = a0*w0/w_/sqrt(1+abs(eps))
    f_ = C_lp * (r/w_)**abs(l)*exp(-1.0*(r/w_)**2)
    f_prime_ = C_lp/w_**3 * exp(-(r/w_)**2) * (r/w_)**(abs(l)-1) * (-2*r**2+w_**2*abs(l))
    
    Ex = a_*f_*g(tau)*cos(l*theta + phi_)
    Ey = -eps*a_*f_*g(tau)*sin(l*theta + phi_)
    Ez = -a_*(f_prime_*g(tau)*cos(theta))*sin(l*theta + phi_) +\
        a_*f_*g(tau)*(l/r*sin(theta)-r*z/(zR*w_**2)*cos(theta))*cos(l*theta+phi_)
        
    Bx = eps*a_*f_*g(tau)*sin(l*theta + phi_)
    By = a_*f_*g(tau)*cos(l*theta + phi_)
    Bz = -a_*(f_prime_*g(tau)*sin(theta))*sin(l*theta+phi_) -\
            a_*f_*g(tau)*(l/r*cos(theta)+r*z/(zR*w_**2)*sin(theta))*cos(l*theta+phi_)
    
    """IF CP POLA"""
    # Ez = -a_*(f_prime_-eps*l/r*f_)*g(tau)*sin((l+eps)*theta+phi_) + a_*f_*(-r*z/(zR*w_**2)*g(tau))*cos((l+eps)*theta+phi_)
    # Bz = a_*(eps*f_prime_-l/r*f_)*g(tau)*cos((l+eps)*theta+phi_) + a_*f_*(-eps*r*z/(zR*w_**2)*g(tau))*sin((l+eps)*theta+phi_)
    return Ex, Ey, Ez, Bx, By, Bz


#==============================
# SPATIAL AND TEMPORAL ENVELOP
#=============================
pwr = 2
c_super=sqrt(2)*scipy.special.gamma((pwr+1)/pwr)

g = g_sin2

#==========================
# LASER PARAMETERS
#==========================
a0 = 2.33
w0 = 2.5*l0
Tp = 12*l0
zR = 0.5*w0**2

eps,l = 0,1
print("a0=",round(a0,2),f"| l={l},s={eps} | w0={w0/l0:.2f} | Tp={Tp/l0:.0f}")

C_lp = np.sqrt(1/factorial(abs(l)))
# #==========================
# # NUMERICAL PARAMETERS
# #==========================
# dt = 0.002
# N = 5_0
# zfoc = 0*l0
# plasma_length = 0*l0
# plasma_pos = 2*l0+1e-12
# TMAX = plasma_pos+1*l0+Tp*1.25#9*l0
# t_range = np.arange(0,TMAX,dt)

# LONGITUDINAL_FIELD = True
# SAVE = False

# rmax = w0*2

# # =========================
# # NUTER INITIAL CONDITIONS
# # =========================
# theta0 = np.random.random(N)*2*pi
# r0 = np.random.random(N)*rmax
# POS = np.zeros((3,N))
# POS[2] = plasma_pos+np.random.random(N)*plasma_length
# POS[0] = r0*cos(theta0)
# POS[1] = r0*sin(theta0)
# VEL = np.zeros((3,N))
# gamma = np.ones(N)

grid = np.arange(-2*w0,2*w0,l0/32)
grid_z = np.arange(-4*l0,5*l0,l0/48)

X,Y,Z = np.meshgrid(grid,grid,grid_z,indexing="ij")

Ex, Ey, Ez, Bx, By, Bz = get_EB_fast(X,Y,Z,t=6*l0)


fig = plt.figure(figsize=plt.figaspect(0.75)*1.25)
ax = fig.add_subplot(projection = '3d',aspect="auto")
ax.set_box_aspect((2, 1, 1))
EM_THRESHOLD = 0.75 #0.75
xc, yc, zc = np.where(np.abs(Ex)>EM_THRESHOLD)

N_eon = 1000
r_max_eon = 2*w0
r_eon = np.random.random(N_eon)
theta_eon = np.random.random(N_eon)*2*pi
x_eon, y_eon = r_max_eon*np.sqrt(r_eon)*np.cos(theta_eon), r_max_eon*np.sqrt(r_eon)*np.sin(theta_eon)

z_eon = 5*l0 + np.random.random(N_eon)*0*l0

scat = ax.scatter(grid_z[zc]/l0,grid[xc]/l0, grid[yc]/l0,s=0.02,c=Ex[xc,yc,zc],cmap="RdBu", label="LG Beam", alpha=0.75)
# plt.colorbar(scat,ax=ax)
ax.scatter(z_eon/l0, x_eon/l0,y_eon/l0, s=3, label="Electrons")
ax.plot((-4.5,5.25),(0,0),(0,0),ls="--",color="k", zorder=100,alpha=0.5)
# ax.grid()
# ax.legend()
ax.set_xlabel("$x/\lambda$",fontsize=13)
ax.set_ylabel("$y/\lambda$",fontsize=13)
ax.set_zlabel("$z/\lambda$",fontsize=13)
ax.set_xlim(-2.75,5.5)
ax.set_ylim(-4,4)
ax.set_zlim(-4,4)

# ax.set_axis_off()
fig.tight_layout()
plt.savefig('3d_plot_no_bg.png', transparent=True)



# azeeeazaezaez

# import plotly.graph_objects as go

# # Create the isosurface plot
# fig = go.Figure(data=go.Isosurface(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=Ex.flatten(),
#     isomin=0.5,  # Adjust according to the desired threshold
#     isomax=Ex.max(),
#     surface_count=5,  # Number of isosurfaces to draw
#     colorscale='Viridis',
# ))

# # # Update the layout
# # fig.update_layout(scene=dict(
# #     xaxis_title='X',
# #     yaxis_title='Y',
# #     zaxis_title='Z',
# # ))

# fig.show()





