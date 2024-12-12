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

# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_A2_HD/opt_a2.0_dx64')


# S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_OPTICAL_NR_HD/opt_base_PML_dx64')

l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt
import math
from numba import njit



def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr


N_part = 1


Int_Ex2_LIST = []
Int_ExM2_LIST = []
Int_AxM2_LIST = []

Int_AxPx_LIST = []
Int_AxPx_over_gammaLIST = []

Int_AzM2_vx_LIST = []
Int_AzM2_gammaP_LIST = []
Lx_amplitude_LIST = []
Int_AxM02_LIST = []
    # print("Ax * px              =",round(Int_AxPx,round_digits))
    # print("Ax^2 *1/(1-vx)       =",round(Int_AzM2_vx,round_digits))
    # print("Ax^2 *sqrt(1+p^2)    =",round(Int_AzM2_gammaP,round_digits))
    # print("Ax^2 *sqrt(1+a0^2/2) =",round(Int_AzM2_gammaA02,round_digits))
    # print("Ax^2 *sqrt(1+a0^2)   =",round(Int_AzM2_gammaA0,round_digits))

for sim in sim_loc_list:
    S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim}')

    T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px","Ex","Ey","Ez","Bx","By","Bz"])

    Ltrans = S.namelist.Ltrans
    Tp = S.namelist.Tp
    w0 = S.namelist.w0
    a0 = S.namelist.a0
    l1 = S.namelist.l1
    track_N_tot = T0.nParticles
    t_range = T0.getTimes()

    track_traj = T0.getData()

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
    vx = px/gamma
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
    g = g_sin2
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
    
    AxM0, AyM0, AzM0 = getE(r[:,Nid],theta[:,Nid],x[0,Nid]-pi/2,t_range)

    

    Int_Ex2 = integrate.simpson(1/gamma[:,Nid]*Ex[:,Nid]**2,x=t_range)
    Int_ExM2 = integrate.simpson(1/gamma[:,Nid]*EzM**2,x=t_range)
    
    Int_ExPx = integrate.simpson(1/gamma[:,Nid]*px[:,Nid]*Ex[:,Nid],x=t_range[:])
    Int_ExMPx = integrate.simpson(1/gamma[:,Nid]*px[:,Nid]*EzM,x=t_range[:])
    Int_AxPx = integrate.simpson(1/gamma[:,Nid]*px[:,Nid]*AzM,x=t_range[:])
    Int_AxM2 = integrate.simpson(1/gamma[:,Nid]*AzM**2,x=t_range[:])
    
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
    
    # plt.figure()
    
    Int_AxPx = integrate.simpson(1/gamma[:,Nid]*AzM*px[:,Nid],x=t_range[:])
    Int_AzM2_vx = integrate.simpson(1/gamma[:,Nid]*AzM**2*1/(1-vx[:,Nid]),x=t_range[:])
    Int_AzM2_gammaP = integrate.simpson(1/gamma[:,Nid]*AzM**2*gamma[:,Nid],x=t_range[:])
    Int_AzM2_gammaA0 = integrate.simpson(1/gamma[:,Nid]*AzM**2*sqrt(1+a0**2),x=t_range[:])
    Int_AzM2_gammaA02 = integrate.simpson(1/gamma[:,Nid]*AzM**2*sqrt(1+a0**2/2),x=t_range[:])
    Int_AxM02 = integrate.simpson(1/gamma[:,Nid]*AzM0**2*sqrt(1+a0**2/2),x=t_range[:])
    Lx_amplitude_LIST.append(np.nanmax(np.abs(Lx_track[-1])))
    
    Int_AxPx_LIST.append(Int_AxPx)
    Int_AzM2_vx_LIST.append(Int_AzM2_vx)
    Int_AzM2_gammaP_LIST.append(Int_AzM2_gammaP)
    
    Int_ExM2_LIST.append(Int_ExM2)
    Int_AxM2_LIST.append(Int_AxM2)
    Int_AxM02_LIST.append(Int_AxM02)

    Int_Ex2_LIST.append(Int_Ex2)
    
    INT_AxPx_over_gamma = integrate.simpson(1/gamma[:,Nid]*AzM*px[:,Nid],x=t_range[:])
    Int_AxPx_over_gammaLIST.append(INT_AxPx_over_gamma)
    print("Ax * px              =",round(Int_AxPx,round_digits))
    print("Ax^2 *1/(1-vx)       =",round(Int_AzM2_vx,round_digits))
    print("Ax^2 *sqrt(1+p^2)    =",round(Int_AzM2_gammaP,round_digits))
    print("Ax^2 *sqrt(1+a0^2/2) =",round(Int_AzM2_gammaA02,round_digits))
    print("Ax^2 *sqrt(1+a0^2)   =",round(Int_AzM2_gammaA0,round_digits))

Int_ExM2_LIST = np.array(Int_ExM2_LIST)
Int_ExM2_LIST = np.array(Int_ExM2_LIST)
Int_AxM02_LIST = np.array(Int_AxM02_LIST)
INT_AxPx_over_gamma = np.array(INT_AxPx_over_gamma)
Lx_amplitude_LIST = np.array(Lx_amplitude_LIST)
Int_Ex2_LIST = np.array(Int_Ex2_LIST)
Int_AxPx_LIST = np.array(Int_AxPx_LIST)
Int_AzM2_vx_LIST = np.array(Int_AzM2_vx_LIST)
Int_AzM2_gammaP_LIST = np.array(Int_AzM2_gammaP_LIST)

r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R, THETA = np.meshgrid(r_range,theta_range)

Tint = 3/8*Tp
z0=5*l0
Lx2_max_model_LIST = []
a0_range_smooth = np.arange(0.1,3,0.01)
for a0 in a0_range_smooth:
    Lx2_max_model_LIST.append(np.max(np.abs(LxEpolar(R,THETA,z0,w0,a0,Tint))))
Lx2_max_model_LIST=np.array(Lx2_max_model_LIST)
    
    
fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel("a0")
# ax.plot(a0_range,Lx_amplitude_LIST,"o",color="blue",label="Smilei Lx",markersize=10)
# ax.plot(a0_range_smooth,Lx2_max_model_LIST/Lx2_max_model_LIST[0],"--",color="blue",label="Lx2 model",lw=2)

ax.plot(a0_range,Int_Ex2_LIST,"k.-",label="Ex^2",lw=1)
ax.plot(a0_range,Int_ExM2_LIST,"k--",label="ExM^2",lw=1)
ax.plot(a0_range,Int_AxM2_LIST,"k-.",label="AxM^2",lw=1)

ax.plot(a0_range,Int_AxPx_LIST,"b.-",label="AxM * px",lw=3)
ax.plot(a0_range,Int_AzM2_vx_LIST,".-", label="AxM^2 *1/(1-vx)")

# ax.plot(a0_range,Int_AzM2_gammaP_LIST/Int_AzM2_gammaP_LIST[0],".-",label="Ax^2 *sqrt(1+p^2)")
ax.plot(a0_range,Int_Ex2_LIST*sqrt(1+a0_range**2/2),".-",label="Ex^2 *$\sqrt{1+a0^2/2}$")
ax.plot(a0_range,Int_AxM2_LIST*sqrt(1+a0_range**2/2),".-",label="AxM^2 *$\sqrt{1+a0^2/2}$")
ax.plot(a0_range,Int_ExM2_LIST*sqrt(1+a0_range**2/2),".-",label="ExM^2 *$\sqrt{1+a0^2/2}$")

ax.legend()
ax.set_xlim(0.9,3.1)

fig.suptitle('Integrated "Intensity" $\int 1/\gamma~A.B~dt$')

# ax.set_xscale("log")
# ax.set_yscale("log")
fig.tight_layout()

fig, ax = plt.subplots()
ax.grid()
ax.set_xlabel("a0")
ax.plot(a0_range,Lx_amplitude_LIST/Lx_amplitude_LIST[0],"o",color="blue",label="Smilei Lx",markersize=10)

ax.plot(a0_range,Int_Ex2_LIST/Int_Ex2_LIST[0],"k.-",label="Ex^2",lw=1)
ax.plot(a0_range,Int_ExM2_LIST/Int_ExM2_LIST[0],"k--",label="ExM^2",lw=1)
ax.plot(a0_range,Int_AxM2_LIST/Int_AxM2_LIST[0],"k-.",label="AxM^2",lw=1)

ax.plot(a0_range,Int_AxPx_LIST/Int_AxPx_LIST[0],"b.-",label="AxM * px",lw=3)
ax.plot(a0_range,Int_AzM2_vx_LIST/Int_AzM2_vx_LIST[0],".-", label="AxM^2 *1/(1-vx)")

ax.plot(a0_range,Int_Ex2_LIST/Int_Ex2_LIST[0]*sqrt(1+a0_range**2/2),".-",label="Ex^2 *$\sqrt{1+a0^2/2}$")
ax.plot(a0_range,Int_AxM2_LIST/Int_AxM2_LIST[0]*sqrt(1+a0_range**2/2),".-",label="AxM^2 *$\sqrt{1+a0^2/2}$")
ax.plot(a0_range,Int_ExM2_LIST/Int_ExM2_LIST[0]*sqrt(1+a0_range**2/2),".-",label="ExM^2 *$\sqrt{1+a0^2/2}$")

ax.legend()
fig.suptitle('Integrated "Intensity" $\int 1/\gamma A.B~dt$ normalized')
ax.set_xlim(0.9,3.1)
# ax.set_xscale("log")
# ax.set_yscale("log")
fig.tight_layout()


