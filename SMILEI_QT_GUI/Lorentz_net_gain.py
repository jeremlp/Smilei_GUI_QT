# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:15:07 2025

@author: Jeremy
"""

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
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
import math

from scipy import integrate,special
from scipy.interpolate import griddata
from numba import njit
import time
from tqdm import tqdm
plt.close("all")

l0 = 2*pi
a0 = 3
w0=2.5*l0
Tp=12*l0
Tmax=Tp*2*1.5
l1=1
t_range = np.arange(5*l0,Tmax,0.1)
t_range_smooth = t_range


x0 = 5*l0



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
t_range_s = np.arange(0,15*l0,0.1)
extent1 = [x_range[0]/l0, x_range[-1]/l0,theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360]
extent2 = [r_range[0]/l0, r_range[-1]/l0, x_range[0]/l0, x_range[-1]/l0]
extent3 = [r_range[0]/l0, r_range[-1]/l0, theta_range[0]/(2*pi)*360, theta_range[-1]/(2*pi)*360]

Y0, Z0 = np.meshgrid(y_range, y_range)
R0 = np.sqrt(Y0**2+Z0**2)
THETA0 = np.arctan2(Z0,Y0)
THETA1, X1 = np.meshgrid(theta_range, x_range)
R2, X2 = np.meshgrid(r_range, x_range)
R3, THETA3 = np.meshgrid(r_range, theta_range)

THETA4, TIME4 = np.meshgrid(theta_range, t_range_s)

r1 = 1.5*l0
Ey, Ez, Ex = getE(r1,THETA1,X1,20*l0)
Y = r1*np.cos(THETA1)
Z = r1*np.sin(THETA1)
Troque_theta_x = - (Y*Ez - Z*Ey)
plt.figure()
plt.imshow(Troque_theta_x,cmap="RdYlBu",aspect="auto",origin="lower")
plt.colorbar()
plt.xlabel("$\Theta$")
plt.ylabel("$x/\lambda$")
plt.title("Torque from Lorentz force $E_\Theta$($\Theta,x$)")

r1 = 1.5*l0
Ey, Ez, Ex = getE(r1,THETA4,5*l0,TIME4)
Y = r1*np.cos(THETA4)
Z = r1*np.sin(THETA4)
Troque_theta_t = - (Y*Ez - Z*Ey)
plt.figure()
plt.imshow(Troque_theta_t,cmap="RdYlBu",aspect="auto",origin="lower")
plt.colorbar()
plt.xlabel("$\Theta$")
plt.ylabel("$t/(c\lambda)$")
plt.title("Torque from Lorentz force $E_\Theta$($\Theta,t$)")


r1 = 1.5*l0
Ey, Ez, Ex = getE(r1,THETA4,5*l0+0.5*TIME4,TIME4)
Y = r1*np.cos(THETA4)
Z = r1*np.sin(THETA4)
Troque_theta_t = - (Y*Ez - Z*Ey)
plt.figure()
plt.imshow(Troque_theta_t,cmap="RdYlBu",aspect="auto",origin="lower")
plt.colorbar()
plt.xlabel("$\Theta$")
plt.ylabel("$t/(c\lambda)$")
plt.title("Torque from Lorentz force $E_\Theta$($\Theta,t,x=0.2*c$)")

def Lorentz_net_gain(coord):
    dr, dtheta, dx = lambda r,theta,x,t:0,lambda r,theta,x,t:0,lambda r,theta,x,t:0

    dr = dr_func
    dx = dx_func
    dtheta = dtheta_func
    max_Lx = []
    min_Lx = []
    mean_Lx = []
    r_range = np.arange(0.5*l0,2*w0,0.1)
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
# plt.xlim(0,2)




def gauss(t,x):
    t_center=1.25*Tp
    c_gauss = sqrt(pi/2)
    return np.exp(-((t-t_center-x)/(Tp*3/8/c_gauss))**2)
plt.figure()
r0 = 0.95*l0
r1 = 1.08*l0
theta0 = pi/4
Ey, Ez, Ex = getE(np.abs(r0+dr_func(r0,theta0,x0,t_range)),theta0,x0,t_range)
y0 = r0*np.cos(theta0)
z0 = r0*np.sin(theta0)
Troque0m = - (y0*Ez - z0*Ey)

Ey, Ez, Ex = getE(np.abs(r1+dr_func(r1,theta0,x0,t_range)),theta0,x0,t_range)
y1 = r1*np.cos(theta0)
z1 = r1*np.sin(theta0)
Troque1m = - (y1*Ez - z1*Ey)

theta0 = -pi/4
Ey, Ez, Ex = getE(np.abs(r0+dr_func(r0,theta0,x0,t_range)),theta0,x0,t_range)
y0 = r0*np.cos(theta0)
z0 = r0*np.sin(theta0)
Troque0p = - (y0*Ez - z0*Ey)

Ey, Ez, Ex = getE(np.abs(r1+dr_func(r1,theta0,x0,t_range)),theta0,x0,t_range)
y1 = r1*np.cos(theta0)
z1 = r1*np.sin(theta0)
Troque1p = - (y1*Ez - z1*Ey)

temp_env = gauss(t_range,x0)
plt.plot(t_range/l0,Troque0p*temp_env,"C0--",label="r=0.95, positive gain")
plt.plot(t_range/l0,Troque1p*temp_env,"C1--",label="r=1.08, negative gain")
plt.grid()
plt.title("Torque seen by 2 electrons at different r")
plt.legend()
t0p = integrate.simpson(Troque0p*temp_env,x=t_range)
t1p = integrate.simpson(Troque1p*temp_env,x=t_range)
print(t0p,t1p)


plt.figure()
plt.plot(t_range[:-1]/l0, integrate.cumulative_simpson(Troque0p*temp_env,x=t_range))
plt.plot(t_range[:-1]/l0, integrate.cumulative_simpson(Troque1p*temp_env,x=t_range))
plt.grid()

plt.title("Lx of 2 electrons at different r")

plt.figure()

t_cross_list = []
r0_range = np.arange(1*l0,1.7*l0,0.001)
for r0 in r0_range:
    dr = r0+dr_func(r0,0,5*l0,t_range)

    t_cross = t_range[np.where(dr<=-0.6*r0)[0][0]]
    t_cross_list.append(t_cross)

plt.plot(r0_range/l0,np.cos(np.array(t_cross_list)),".-")
plt.grid()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$\cos(t_c)$")
plt.title("Phase $\propto \cos(t_c)$")