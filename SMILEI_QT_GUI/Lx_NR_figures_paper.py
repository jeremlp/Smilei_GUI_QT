# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:02:44 2024

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
import matplotlib.ticker as ticker

plt.close("all")

a0_requested = 0.1

sim_loc_list_12 = ["SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12", #gauss_a1_Tp12_dx128_AM8
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


sim_path = "SIM_OPTICAL_GAUSSIAN/gauss_a0.1_Tp12_dx128_AM8"
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_path}')

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')

l0=2*np.pi
T0 = S.TrackParticles("track_eon_full", axes=["x","y","z","py","pz","px"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1


track_N_tot = T0.nParticles
t_range = T0.getTimes()
t_range_smooth = np.arange(0,t_range[-1],0.01)
track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1

if "AM" in sim_path:
    track_r_center = 0
else:
    track_r_center = Ltrans/2

x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-track_r_center
z = track_traj["z"][:,::N_part] -track_r_center
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

# Ey = track_traj["Ey"][:,::N_part]
# Ez = track_traj["Ez"][:,::N_part]
# Ex = track_traj["Ex"][:,::N_part]
# By = track_traj["By"][:,::N_part]
# Bz = track_traj["Bz"][:,::N_part]
# Bx = track_traj["Bx"][:,::N_part]


r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
vx = px/gamma

tmax = 120

def min_max(X,Y,dr_av=0.15):
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
# @njit
# def Ftheta_V2_O3(r,theta,z):
#     #-1 bcs Ftheta_V2_O3 uses IntBz instead of IntEz, so need to inverse sign
#     expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
#           a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) +
#                           4 * (2 * r**2 * z * cos(theta)**2 +
#                                2 * r**4 * cos(theta)**3 * sin(theta) -
#                                2 * r**2 * z * sin(theta)**2 +
#                                2 * r**4 * cos(theta) * sin(theta)**3))) /
#          (2 * r * (w0**4 + 4 * z**2)**3))
#     return expr
# @njit
# def Ftheta_V2_O3Cart(x,y,z):
#     #/!\ USES Z AS PROPAGATION DIRECTION /!\

#     #-1 bcs Ftheta_V2_O3 uses IntBz instead of IntEz, so need to inverse sign

#     r = sqrt(x**2+y**2)
#     theta = arctan2(y,x)
#     expr = -1 * -((exp(-((2 * w0**2 * (r**2 * cos(theta)**2 + r**2 * sin(theta)**2)) / (w0**4 + 4 * z**2))) *
#           a0**2 * w0**6 * (-8 * r**2 * w0**2 * cos(theta) * sin(theta) +
#                           4 * (2 * r**2 * z * cos(theta)**2 +
#                                2 * r**4 * cos(theta)**3 * sin(theta) -
#                                2 * r**2 * z * sin(theta)**2 +
#                                2 * r**4 * cos(theta) * sin(theta)**3))) /
#          (2 * r * (w0**4 + 4 * z**2)**3))
#     return expr

# @njit
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
def Torque_V2_O3(r,theta,z):
    """ Torque = r*Ftheta (r^2 appear instead of r) """
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r**2 * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3


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

Nid=0
x_pos = 5*l0
x0 = x_pos
r0 = r[0,Nid]
theta0 = theta[0,Nid]
x_range = np.arange(-1*w0,1*w0,0.1)
X,Y = np.meshgrid(x_range,x_range)
extent = [x_range[0]/l0,x_range[-1]/l0,x_range[0]/l0,x_range[-1]/l0]

# plt.figure()
# plt.scatter(r[0]/l0,Lx_track[-1],s=1)

r_range = np.arange(0,2*w0,0.05)
theta_range = np.arange(0,2*pi,pi/32)
R,THETA = np.meshgrid(r_range,theta_range)
Lx_max_model = np.max(LxEpolar_V2_O3(R,THETA,x0,w0,a0,3/8*Tp),axis=0)


#==================================================
r_requested = 1.*l0
Nid = np.where(np.abs(r[0]-r_requested)==np.min(np.abs(r[0]-r_requested)))[0][0]
#==================================================

plt.figure()
plt.plot(t_range/l0,r[:,Nid]/l0,".-")
# plt.plot(t_range/l0, r[0,Nid]+dr_short_time(r[0,Nid], t_range))
plt.plot(t_range/l0, np.abs(r[0,Nid]+dr(r[0,Nid],t_range,x_pos))/l0,label="r+$\Delta r(t)$")
plt.plot(t_range/l0, np.abs(r[0,Nid]+dr_Relat_mean(r[0,Nid],t_range,x_pos))/l0,label="r+$\Delta r(t)$ relat 1/$\sqrt{1+(f(r)*a_0)^2/2 + 1/16(f(r)*a0)^4}$")
plt.plot(t_range/l0, np.abs(r[0,Nid]+dr_Relat(r[0,Nid],t_range,x_pos))/l0,label="r+$\Delta r(t)$ relat 1/$\sqrt{1+(f(r)*a_0)^2 + 1/4(f(r)*a0)^4}$")



plt.legend()
plt.grid()
plt.xlabel("t/t0")
plt.title(f"Radial displacement models\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")

plt.figure()

plt.plot(t_range/l0,1/gamma[:,Nid]*r[:,Nid]*Ftheta_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*gauss(t_range,x[:,Nid])**2,".-",label="r*Ftheta_V2_O3 exact")
# plt.plot(t_range/l0,(r0) * Ftheta_V2_O3(r0, theta0, x_pos)*gauss(t_range,x[:,Nid])**2,label="r0*Ftheta_V2_O30 NR")
plt.plot(t_range/l0,1/sqrt(1+(f(r0,x0)*a0)**2/2+ 1/16*(f(r0,x0)*a0)**4) * np.abs(r0+dr(r0,t_range,x_pos)) * Ftheta_V2_O3(np.abs(r0+dr(r0,t_range,x_pos)), theta0, x0)*gauss(t_range,x[:,Nid])**2,label="(r+dr)*Ftheta_V2_O3 dr")
plt.grid()
plt.legend()

gamma42 = sqrt(1+(f(r0,x0)*a0)**2/2+ 1/16*(f(r0,x0)*a0)**4)
a = integrate.simpson(1/gamma42 *r[:,Nid]*Ftheta_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*gauss(t_range,x[:,Nid])**2,x=t_range)
b = integrate.simpson(1/gamma42 * np.abs(r0+dr(r0,t_range,x_pos)) * Ftheta_V2_O3(np.abs(r0+dr(r0,t_range,x_pos)), theta0, x0)*gauss(t_range,x[:,Nid])**2, x=t_range)
print("Ftheta_V2_O3 integration (exact, r+dr):",a,b)
# azeazeeazzaeazzzzzzzzzzzzzzzzzzzzzzzzzzzzzezeazzeaaezeza

def Lx4_distrib(dr_func):
    t_range_lx = t_range_smooth#np.arange(0,t_range[-1],dt)
    temp_env = gauss(t_range_lx,x_pos)
    distrib_Lx_list = []
    for r in r_range:
        Lx_theta_list = []
        for theta in np.arange(0,2*pi,pi/32):
            LxR_distrib = integrate.simpson(Torque_V2_O3(np.abs(r+dr_func(r,t_range_lx,x_pos)), theta, x_pos)*temp_env**2, x=t_range_lx)
            Lx_theta_list.append(LxR_distrib)
        distrib_Lx_list.append(np.max(Lx_theta_list))
    return np.array(r_range),np.array(distrib_Lx_list)

def Lx_distrib_FullMotion():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * Torque_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)


def Lx_distrib_FullMotion_FV2O5():
    t_range_lx = t_range#np.arange(0,t_range[-1],dt)
    distrib_r_list = []
    distrib_Lx_list = []
    for Nid in range(len(x[0])):
        temp_env = gauss(t_range,x[:,Nid])
        LxR_distrib = integrate.simpson(1/gamma[:,Nid] * Torque_V2_O3(r[:,Nid], theta[:,Nid], x[:,Nid])*temp_env**2, x=t_range_lx)
        distrib_Lx_list.append(LxR_distrib)
        distrib_r_list.append(r[0,Nid])
    return np.array(distrib_r_list),np.array(distrib_Lx_list)

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

# plt.colorbar(myplot, format=ticker.FuncFormatter(fmt))
#%% Plot Radial distribution

fig1 = plt.figure(figsize=(9,5))
# plt.scatter(r[0]/l0,Lx_track[-1],s=4,alpha=1)
a_range, lower_Lx, upper_Lx = min_max(r[0],Lx_track[-1])
plt.fill_between(a_range/l0, lower_Lx, upper_Lx,color="lightblue")
plt.plot(a_range/l0,lower_Lx,"C0",lw=2)
plt.plot(a_range/l0,upper_Lx,"C0",lw=2, label=f"Smilei")

Lx_max_model = np.max(LxEpolar_V2_O5(R,THETA,x0,w0,a0,3/8*Tp),axis=0)
COEF = sqrt(1+(a0*f(r_range,x_pos))**2+ 1/4*(a0*f(r_range,x_pos))**4)
plt.plot(r_range/l0,Lx_max_model,"k-",lw=2)
plt.plot(r_range/l0,-Lx_max_model,"k-", lw=2,label="Model $L_x^{NR}$")
plt.grid()
plt.legend()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("$L_x$")
# plt.title(f"$L_x$ distribution comparison between Smilei and model\n($a_0={a0},Tp={Tp/l0:.0f}t_0,w_0=2.5\lambda$)")
plt.tight_layout()
plt.xlim(0,5)
# aezaezzea


fig1.tight_layout()


#%% Plot Transverse distribution

dx_interp = 0.05*l0
y_range = np.arange(-2*w0,2*w0,dx_interp)
Y, Z = np.meshgrid(y_range,y_range, indexing='ij')
THETA = np.arctan2(Z,Y)
R = np.sqrt(Y**2+Z**2)
extent = [y_range[0]/l0,y_range[-1]/l0,y_range[0]/l0,y_range[-1]/l0]
Lx_smilei_interp = griddata(np.c_[y[0],z[0]], Lx_track[-1], (Y, Z), method='cubic')
# Lx_smilei_interp[np.isnan(Lx_smilei_interp)] = 0.0

Lx_NR = LxEpolar_V2_O3(R,THETA,x0,w0,a0,3/8*Tp)

fig2, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
im1 = ax1.imshow(Lx_smilei_interp, extent=extent, cmap="RdYlBu")
im2 = ax2.imshow(Lx_NR, extent=extent, cmap="RdYlBu")


fig2.colorbar(im2,ax=ax2,pad=0.01) #,format=ticker.FuncFormatter(fmt)
ax1.set_xlabel("$y_0/\lambda$")
ax1.set_ylabel("$z_0/\lambda$")
ax2.set_xlabel("$y_0/\lambda$")
ax1.set_title("Smilei $L_x$")
ax2.set_title("Model $L_x^{NR}$")

circ1 = plt.Circle((0, 0), w0/l0, color='k', fill=False,ls="--",label="$r=w_0$")
circ2 = plt.Circle((0, 0), w0/l0, color='k', fill=False,ls="--")
ax1.add_patch(circ1)
ax2.add_patch(circ2)
ax1.legend()

ax2.set_yticks([])  # Remove x-axis ticks
ax2.yaxis.set_visible(False)
# fig2.colorbar(im2, ax=ax2,pad=0.01) #,format=ticker.FuncFormatter(fmt)

fig2.tight_layout()


SAVE = False
if SAVE:
    
    fig1.savefig(f"{os.environ['SMILEI_QT']}/figures/_PAPER_OAM_/Non-relat_model/Lx_NR_radial_a{a0}.png")
    fig2.savefig(f"{os.environ['SMILEI_QT']}/figures/_PAPER_OAM_/Non-relat_model/Lx_smilei_transverse_a{a0}.png")
    fig3.savefig(f"{os.environ['SMILEI_QT']}/figures/_PAPER_OAM_/Non-relat_model/Lx_NR_transverse_a{a0}.png")