# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 10:35:55 2024

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



zR = 0.5*w0**2
eps,l1 = 0,1

C_lp = np.sqrt(1/math.factorial(abs(l1)))*sqrt(2)**abs(l1)
c_gauss = sqrt(pi/2)
def g_gauss(tau):
    return  exp( -((tau-1.25*Tp)/(Tp*3/8/c_gauss))**2) 
def g_sin2(tau):
    return  sin(pi*tau/Tp)**2*(tau>0)*(tau<Tp)   

# g = g_sin2
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


r_target=w0/sqrt(2)
Nid = np.where(np.abs(r[0]-r_target)==np.min(np.abs(r[0]-r_target)))[0][0]


ExM, EyM, EzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid],t_range)
AxM, AyM, AzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid]-pi/2,t_range)


plt.figure()
plt.plot(t_range,Ey[:,Nid],".-",label="Smilei Ey")
plt.plot(t_range,ExM,".-",label="Model Ey")
plt.grid()
plt.legend()

plt.figure()
plt.plot(t_range,Ex[:,Nid],".-",label="Smilei Ex")
plt.plot(t_range,EzM,".-",label="Model Ex")
plt.grid()
plt.legend()

plt.figure()
plt.plot(t_range[:],Ex[:,Nid],".-",label="Smilei Ex")
plt.plot(t_range,AzM,".-",label="Model Ax")
plt.grid()
plt.legend()


# ExM, EyM, EzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid],t_range)
# AxM, AyM, AzM = getE(r[:,Nid],theta[:,Nid],x[:,Nid]-pi/2,t_range)

# AxM0, AyM0, AzM0 = getE(r[:,Nid],theta[:,Nid],x[0,Nid]-pi/2,t_range)



Int_Ex2 = integrate.simpson(Ex[:,Nid]**2,x=t_range)
Int_ExM2 = integrate.simpson(EzM**2,x=t_range)

Int_ExPx = integrate.simpson(px[:,Nid]*Ex[:,Nid],x=t_range[:])
Int_ExMPx = integrate.simpson(px[:,Nid]*EzM,x=t_range[:])
Int_AxPx = integrate.simpson(px[:,Nid]*AzM,x=t_range[:])
Int_AxM2 = integrate.simpson(AzM**2,x=t_range[:])

# print(f"Ax using shift of {pi/(t_range[3]-t_range[0])}")
round_digits = 3
print("="*15)
print(f"a0 = {a0}")
print("="*15)
print("Ex^2   =",round(Int_Ex2,round_digits))
print("ExM^2  =",round(Int_ExM2,round_digits))
print("AxM^2  =",round(Int_AxM2,round_digits))

print("-"*10)

print("Ex*px  =",round(Int_ExPx,round_digits))
print("ExM*px =",round(Int_ExMPx,round_digits))
print("-"*10)

print("AxM*px       =",round(Int_AxPx,round_digits))
# print("ExM^2*gamma  =",round(Int_ExM2*sqrt(1+a0**2),round_digits))
# print("ExM^2*gamma2 =",round(Int_ExM2*sqrt(1+a0**2/2),round_digits))
print("AxM^2*gamma  =",round(Int_AxM2*sqrt(1+a0**2),round_digits))
print("AxM^2*gamma2 =",round(Int_AxM2*sqrt(1+a0**2/2),round_digits))

print("="*15,"\n")

# print("Ax*px =",round(Int_AxPx,5))

# ==========
# Ex^2 = 0.23641
# ExM^2 = 0.20837
# AxM^2 = 0.22933
# ----------
# Ex*px = 0.07706
# ExM*px = 0.22196
# AxM*px = 0.50103
# ExM^2*gamma = 0.46592
# ExM^2*gamma_bis = 0.3609