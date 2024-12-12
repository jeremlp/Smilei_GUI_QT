# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:55:11 2024

@author: Jeremy
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos, sin, arctan2, exp, sqrt
import math
l0=2*pi


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
from scipy import integrate
from scipy import special

plt.close("all")

a0_requested = 2.33

sim_loc_list = ["SIM_OPTICAL_NR_HD/opt_base_PML_dx64",
                "SIM_OPTICAL_A0.3_HD/opt_a0.3_dx48",
                "SIM_OPTICAL_A1_HD/opt_a1.0_dx64",
                "SIM_OPTICAL_A1.5_HD/opt_a1.5_dx48",
                "SIM_OPTICAL_A2_HD/opt_a2.0_dx64",
                "SIM_OPTICAL_A2.33_HD/opt_a2.33_dx48",
                "SIM_OPTICAL_A2.5_HD/opt_a2.5_dx48",
                "SIM_OPTICAL_A3_HD/opt_a3.0_dx32"]
a0_range = np.array([0.1,0.3,1,1.5,2,2.33,2.5,3])

a0_sim_idx = np.where(a0_range==a0_requested)[0][0]
sim_path = sim_loc_list[a0_sim_idx]

S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_GAUSSIAN/gauss_a2_Tp12')
# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')


# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","Bx","By","Bz"])

Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
l1 = S.namelist.l1


track_N_tot = T0.nParticles
t_range = T0.getTimes()

track_traj = T0.getData()

print(f"RESOLUTION: l0/{l0/S.namelist.dx}")

N_part = 1



x = track_traj["x"][:,::N_part]
y = track_traj["y"][:,::N_part]-Ltrans/2
z = track_traj["z"][:,::N_part] -Ltrans/2
py = track_traj["py"][:,::N_part]
pz = track_traj["pz"][:,::N_part]
px = track_traj["px"][:,::N_part]

Ey = track_traj["Ey"][:,::N_part]
Ez = track_traj["Ez"][:,::N_part]
Ex = track_traj["Ex"][:,::N_part]
By = track_traj["By"][:,::N_part]
Bz = track_traj["Bz"][:,::N_part]
Bx = track_traj["Bx"][:,::N_part]


r = np.sqrt(y**2+z**2)
theta = np.arctan2(z,y)
pr = (y*py + z*pz)/r
Lx_track =  y*pz - z*py
gamma = sqrt(1+px**2+py**2+pz**2)
x_pos = x[0,0]

tmax = 120

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


def dr_long_time(r,t):
    pr_end = -a0**2/4*f_squared_prime(r, x_pos)*3/8*Tp
    return pr_end*(t-Tp-x_pos)

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

def pr_model_old(r):
    return -a0**2/4*f_squared_prime(r, x_pos)* 3/8*Tp
            # self.track_displace_pr_model, = self.ax4_displace.plot(r_range/l0, -self.a0**2/4*self.f_squared_prime(r_range,x_pos-self.xfoc)*3/8*self.Tp,"r--",label="Model")

def pr_model(r,t):
    return -a0**2/4*f_squared_prime(r, x_pos) * gauss2_int(t,x)

def pr_model_relat(r,t):
    return -1/gamma[:,Nid]*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int(t,x_pos)
def pr_model_relat_end(r):
    return -1/gamma[-1,Nid]*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int(50*l0,x_pos)

def dr_model_relat(r,t):
    return -1/gamma[:,Nid]*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int_int(t,x_pos)
def dr_model_relat_a0(r,t):
    return -1/sqrt(1+f(r,x_pos)*a0**2)*a0**2/4*f_squared_prime(r, x_pos)* gauss2_int_int(t,x_pos)
def dr_model(r,t):
    return -a0**2/4*f_squared_prime(r, x_pos)* gauss2_int_int(t,x_pos)
r_target=3.3*l0
Nid = np.where(np.abs(r[0]-r_target)==np.min(np.abs(r[0]-r_target)))[0][0]
# interaction_delay = (1.25*Tp/l0-Tp/l0/2+5)*l0

r0 = r[0,Nid]
plt.figure()
plt.plot(t_range/l0,pr[:,Nid],".-",label="Smilei")
plt.plot(t_range/l0,pr_model(r0,t_range),"k--",label="pr_model")
plt.plot(t_range/l0,1/sqrt(1+(a0*f(r0,x_pos))**2/2)*pr_model(r0,t_range),"g--",label="pr_model/<gamma>")
plt.plot(t_range/l0,pr_model_relat(r0,t_range),"r--",label="pr_model/gamma(t)")
plt.xlabel("$t/t_0$")
plt.ylabel("pr")
plt.grid()
plt.legend()



# plt.figure()
# plt.plot(t_range,gauss2_int_int(t_range,x_pos))
# plt.grid()

plt.figure()
plt.plot(t_range/l0,r[:,Nid],".-",label="Smilei")
plt.plot(t_range/l0,r0+dr_model(r0,t_range),"C1--",label="dr_model")
plt.plot(t_range/l0,r0+dr_model_relat(r0,t_range),"C2--",label="dr_model relat")
plt.plot(t_range/l0,r0+dr_model_relat_a0(r0,t_range),"C3-",label="dr_model relat a0")
plt.xlabel("$t/t_0$")
plt.ylabel("$\Delta r$")
plt.grid()
plt.legend()


aezaezaezae


r_range = np.arange(0,2*w0,0.1)

plt.figure()
x_pos = 5*l0
plt.scatter(r[0]/l0,pr[-1],s=1)
plt.plot(r_range/l0,pr_model(r_range,40*l0),"k--",label="pr_model")

plt.plot(r_range/l0,1/sqrt(1+(a0*f(r0,6.5*l0))**2/2)*pr_model(r_range,40*l0),"g--",label="pr_model/<gamma>")

plt.plot(r_range/l0,pr_model_relat_end(r_range),color="red",ls="--",label="pr_model/gamma(t)")

# plt.plot(r_range/l0,1/sqrt(1+(a0*f(r0,x_pos))**2/2)*pr_model_old(r_range),"g--")
plt.legend()
plt.grid()
plt.xlabel("$r_0/\lambda$")
plt.ylabel("pr")

# pr_list = []
# for r in r_range:
#     pr_model(r,)

# plt.figure()
# plt.plot(t_range/l0,pr[:,Nid],".-",label="Smilei")
# plt.plot(t_range/l0,pr_model(r0,t_range-interaction_delay),"k--",label="Model pr")
# plt.plot(t_range/l0,pr_model_relat(r0,t_range-interaction_delay),"r--",label="Model pr_relat")
# plt.plot(t_range/l0,1/sqrt(1+(a0*f(r0,x_pos))**2/2)*pr_model(r0,t_range-interaction_delay),"g--",label="Model pr/gamma(r)")

# plt.grid()
# plt.legend()


# t_max = -150

# plt.figure()
# plt.plot(pr[:t_max,Nid],px[:t_max,Nid],".-")
# plt.scatter(pr[0,Nid],px[0,Nid],s=100,color="red")
# plt.grid()

# plt.figure()
# plt.plot(r[:t_max,Nid],x[:t_max,Nid],".-")
# plt.scatter(r[0,Nid],x[0,Nid],s=100,color="red")
# plt.grid()