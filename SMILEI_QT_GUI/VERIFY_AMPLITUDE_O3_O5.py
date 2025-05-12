# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:53:09 2024

@author: jerem
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
module_dir_happi = f"{os.environ['SMILEI_SRC']}"
sys.path.insert(0, module_dir_happi)
import happi
from scipy.optimize import curve_fit
plt.close("all")
l0=2*np.pi
from numpy import exp, sin, cos, arctan2, pi, sqrt

Tp_requested = 12

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


def w(z):
    zR = 0.5*w0**2
    return w0*np.sqrt(1+(z/zR)**2)
def f(r,z):
    l1=1
    return (r*sqrt(2)/w(z))**abs(l1)*np.exp(-(r/w(z))**2)
def LxEpolar(r,Theta,z,w0,a0,Tint):
    expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
        -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
        4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
        4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
        4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
        r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
    return expr
def Ftheta_V2_O3(r,theta,z,a0):
    return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
            (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3

def Ftheta_V2_O5(r,theta,z,a0):
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
def LxEpolar_V2_O3(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O3(r,theta,z,a0)

def LxEpolar_V2_O5(r,theta,z,w0,a0,Tint):
    return Tint*r*Ftheta_V2_O5(r,theta,z,a0)


N_part = 1
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_list_12[0]}')
T0 = S.TrackParticles("track_eon", axes=["x","y","z","py","pz","px"])
Ltrans = S.namelist.Ltrans
Tp = S.namelist.Tp
w0 = S.namelist.w0
a0 = S.namelist.a0
    
# Lx_amplitude_list_12 = []
# for k,sim_loc_12 in enumerate(sim_loc_list_12[:]):
#     S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/{sim_loc_12}')
#     T0 = S.TrackParticles("track_eon_full", axes=["x","y","z","py","pz","px"])

#     Ltrans = S.namelist.Ltrans
#     Tp = S.namelist.Tp
#     w0 = S.namelist.w0
#     a0 = S.namelist.a0
#     track_N_tot = T0.nParticles
#     t_range = T0.getTimes()
    
#     track_traj = T0.getData()

#     print(f"RESOLUTION: l0/{l0/S.namelist.dx}")
#     if "AM" in sim_loc_12:
#         track_r_center = 0
#     else:
#         track_r_center = Ltrans/2
        
#     x = track_traj["x"][:,::N_part]
#     track_N = x.shape[1]
    
#     y = track_traj["y"][:,::N_part]-track_r_center
#     z = track_traj["z"][:,::N_part] -track_r_center
#     py = track_traj["py"][:,::N_part]
#     pz = track_traj["pz"][:,::N_part]
#     px = track_traj["px"][:,::N_part]
#     r = np.sqrt(y**2+z**2)
#     Lx_track =  y*pz - z*py
#     Lx_amplitude_list_12.append(np.nanmax(Lx_track[-1]))


Lx_amplitude_list_12 = np.loadtxt(f"{os.environ['SMILEI_QT']}/data/amplitude_smilei/Lx_amplitude_list_12.txt")


# eazazeazeaez


a0_range_smooth = np.arange(0.1,4.5+0.1,0.01)


r_range = np.arange(0,2*w0,0.1)
theta_range = np.arange(0,2*pi,pi/16)
R, THETA = np.meshgrid(r_range,theta_range)

x0=5*l0


# Lx_max_model_list_12 = []
# Lx_max_model_list_12_O3 = []
Lx_max_model_list_12_O5 = []
Lx_max_model_list_12_O3_GAMMA = []
Lx_max_model_list_12_O5_GAMMA = []
Lx_max_model_list_12_O5_GAMMA_MEAN = []
Lx_max_model_list_12_GAMMA_perp = []
Lx_max_model_list_12_O5_GAMMA_MEAN_squared = []

Lx_max_model_list_12_O5_GAMMA_MEAN_2 = []

Tint = 3/8*12*l0
for a0 in a0_range_smooth[:]:

    # Lx_max_model_list_12.append(np.max(LxEpolar(R,THETA,5*l0,2.5*l0,a0,3/8*12*l0)))
    
    # Lx_max_model_list_12_O3.append(np.max(LxEpolar_V2_O3(R,THETA,5*l0,2.5*l0,a0,3/8*12*l0)))

    
    Lx_max_model_list_12_O5.append(np.max(LxEpolar_V2_O5(R,THETA,5*l0,2.5*l0,a0,Tint)))
    # Lx_max_model_list_12_O3_GAMMA.append(np.max(np.sqrt(1 + (f(R,x0)*a0)**2 + 1/4*(f(R,x0)*a0)**4)*LxEpolar_V2_O3(R,THETA,x0,2.5*l0,a0,3/8*12*l0)))
    # Lx_max_model_list_12_GAMMA_perp.append(np.max(np.sqrt(1 + (f(R,x0)*a0)**2)*LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,Tint)))

    Lx_max_model_list_12_O5_GAMMA.append(np.max(np.sqrt(1 + (f(R,x0)*a0)**2 + 1/4*(f(R,x0)*a0)**4)*LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,Tint)))
    # Lx_max_model_list_12_O5_GAMMA.append(np.max(LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,Tint)))

    Lx_max_model_list_12_O5_GAMMA_MEAN.append(np.max(np.sqrt(1 + (f(R,x0)*a0)**2/2 + 1/16*(f(R,x0)*a0)**4)*LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,3/8*12*l0)))
    
    
    # Lx_max_model_list_12_O5_GAMMA_MEAN_squared.append(np.max((1 + (f(R,x0)*a0)**2/2 + 1/16*(f(R,x0)*a0)**4)*LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,3/8*12*l0)))
    # Lx_max_model_list_12_O5_GAMMA_MEAN_2.append(np.max(1.5*np.sqrt(1 + (f(R,x0)*a0)**2/2 + 1/16*(f(R,x0)*a0)**4)*LxEpolar_V2_O5(R,THETA,x0,2.5*l0,a0,3/8*12*l0)))
    
    
# Lx_max_model_list_12 = np.array(Lx_max_model_list_12)
# Lx_max_model_list_12_O3 = np.array(Lx_max_model_list_12_O3)
Lx_max_model_list_12_O5 = np.array(Lx_max_model_list_12_O5)
Lx_max_model_list_12_O3_GAMMA = np.array(Lx_max_model_list_12_O3_GAMMA)
Lx_max_model_list_12_O5_GAMMA = np.array(Lx_max_model_list_12_O5_GAMMA)
Lx_max_model_list_12_O5_GAMMA_MEAN = np.array(Lx_max_model_list_12_O5_GAMMA_MEAN)
# Lx_max_model_list_12_GAMMA_perp = np.array(Lx_max_model_list_12_GAMMA_perp)
Lx_max_model_list_12_O5_GAMMA_MEAN_squared = np.array(Lx_max_model_list_12_O5_GAMMA_MEAN_squared)
Lx_max_model_list_12_O5_GAMMA_MEAN_2 = np.array(Lx_max_model_list_12_O5_GAMMA_MEAN_2)

a0r = a0_range_smooth
r1 = w0*sqrt(2* (2 - sqrt(2)))/2


plt.figure()
# plt.plot(a0_range_smooth,Lx_max_model_list_12_O5,"k-",label="Max |$L_x^{NR}$|", lw=2)
plt.plot(a0_range_smooth,Lx_max_model_list_12_O5,"k--",label="Max |$L_x^{NR}$|", lw=2)
gamma_max = np.sqrt(1 + (f(r1,x0)*a0_range_smooth)**2 + 1/4*(f(r1,x0)*a0_range_smooth)**4)
# plt.plot(a0_range_smooth,gamma_max*Lx_max_model_list_12_O5,"C4--",label="Max g*|$L_x^{NR}$|", lw=2)


k = f(1.5*l0,5*l0)
# plt.plot(a0_range_smooth,Lx_max_model_list_12_GAMMA_perp,
#           "-.",color="C2",label="Max $|\gamma_{max,\perp}~L_x^{NR}|$", lw=2)


# plt.plot(a0_range_smooth,Lx_max_model_list_12_O5 * np.sqrt(1+(k*a0r)**2/2),
#           "-.",color="C2",label="Max $\sqrt{1+(f*a_0)^2/2}*|L_x^{NR}|$", lw=2)


plt.plot(a0r,Lx_max_model_list_12_O5_GAMMA,
          "-", color="r",label="Max $|\gamma_{max}~L_x^{NR}|$", lw=2)

plt.plot(a0r,Lx_max_model_list_12_O5_GAMMA_MEAN,
          "--", color="C2",label="Max $|<\gamma>~L_x^{NR}|$", lw=2)

# plt.plot(a0r,Lx_max_model_list_12_O5_GAMMA_MEAN_2,
#           "--", color="C3",label="Max $2<\gamma>~L_x^{NR}|$", lw=2)

# plt.plot(a0r,Lx_max_model_list_12_O5_GAMMA_MEAN_squared,
#           "--", color="C4",label="Max $<\gamma>^2~|L_x^{NR}|$", lw=2)

# plt.plot(a0_range_smooth,Lx_max_model_list_12_GAMMA_base,
#          "-", color="C0", lw=2)



expr_rmin = (np.sqrt(2) - 1) * np.exp(np.sqrt(2) - 2) * (a0_range_smooth**2 * Tint / w0**2)
# COEF = sqrt(1+(a0_range_smooth*f(r1,0))**2+ 1/4*(a0_range_smooth*f(r1,0))**4)
# plt.plot(a0_range_smooth,COEF*expr_rmin,"C4--",label="Analytical $L_{x,max}$ with $\gamma_{max}(x=0)$")

COEF = sqrt(1+(a0_range_smooth*f(r1,x0))**2+ 1/4*(a0_range_smooth*f(r1,x0))**4)
# plt.plot(a0_range_smooth,COEF*expr_rmin,"C2--",label="Analytical $L_{x,max}$")



# plt.plot(a0_range_smooth,COEF*LxEpolar_V2_O3(r1,-pi/4,0,w0,a0_range_smooth,Tint),"C4--",label="Analytical with LxO3")
# plt.plot(a0_range_smooth,COEF*LxEpolar_V2_O5(r1,-pi/4,0,w0,a0_range_smooth,Tint),"C4-.",label="Analytical with LxO5")


plt.plot(a0_range_12,Lx_amplitude_list_12,"o",c="C0",label="Smilei",markersize=10)

plt.grid()
plt.legend(fontsize=12)
plt.xlabel("$a_0$",fontsize=16)
plt.title(f"$L_x$ amplitude scaling with $a_0$\n$T_p$=12$~t_0;$ $w_0$={w0/l0:.1f}λ")
plt.ylabel("max $|L_x|$",fontsize=16)
plt.xlim(0.8,4.6)
plt.tight_layout()


