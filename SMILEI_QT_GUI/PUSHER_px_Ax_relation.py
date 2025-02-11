# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:43:11 2025

@author: Jeremy
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
from scipy.signal import savgol_filter
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
@njit
def g_sin3(tau):
    return sin(pi*tau/Tp)**3 * (tau<Tp) * (tau>0)
@njit
def f_nuter(r,z):
    return C_lp * (r/w(z))**abs(l)*exp(-1.0*(r/w(z))**2)
@njit
def f_vlad(r,z):
    """ADDED A 1/2 IN EXP """
    return C_lp * (r/w(z))**abs(l)*exp(-0.5*(r/w(z))**2)
@njit
def f_prime_nuter(r,z):
    return C_lp/w(z)**3 * exp(-(r/w(z))**2) * (r/w(z))**(abs(l)-1) * (-2*r**2+w(z)**2*abs(l))
@njit
def f_prime_vlad(r,z):
    return C_lp/w(z)**3 * exp(-0.5*(r/w(z))**2) * (r/w(z))**(abs(l)-1) * (-r**2+w(z)**2*abs(l))

@njit
def w(z):
    return w0*sqrt(1+(z/zR)**2)
@njit
def a(z):
    return a0*w0/w(z)/sqrt(1+abs(eps))
@njit
def Rc(z):
    return z*(1+(zR/z)**2)
@njit
def phi(z,r,t):
    return z - t + ( r**2/(2*Rc(z)) ) - (abs(l)+1)*arctan2(z,zR)



@njit(parallel=True)
def get_EB_fast(x,y,z,t):
    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    tau = t-z
    w_ = w0*sqrt(1+(z/zR)**2)
    Rc_ = z*(1+(zR/z)**2)
    phi_ = z - t + ( r**2/(2*Rc_) ) - (abs(l)+1)*arctan2(z,zR) + pi
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


@njit
def cross_axis0(A, B):
    result = np.empty_like(A)
    for i in range(A.shape[1]):
        result[:, i] = np.cross(A[:, i], B[:, i])
    return result

# @profile
@njit
def RELAT_PUSH(E, B, dt, POS, VEL, gamma):
    """RELATIVISTIC :
    https://www.dropbox.com/scl/fo/0wxmu8o29tvczsdd7h53a/h?dl=0&preview=vay_relativistic.pdf&rlkey=oxxxxb6l2c1467nlmshskv2t2 """
    # =========================
    #STEP 1: half step u_i+1/2
    # =========================
    u_i = gamma*VEL
    u_half = u_i + 0.5*rs*dt*(E + cross_axis0(VEL, B))

    # =========================
    #STEP 2: half step u_i
    # =========================
    uprim = u_half +  0.5*rs*dt*E
    gamma_prim = np.sqrt(1 + np.sum(uprim**2,axis=0))
    tau = 0.5*rs*dt*B
    sigma = gamma_prim**2 - np.sum(tau**2,axis=0)
    u_star = uprim[0]*tau[0] + uprim[1]*tau[1] + uprim[2]*tau[2]#np.dot(uprim, tau,axis=0)

    new_gamma = np.sqrt(0.5*(sigma + np.sqrt(sigma**2 + 4*(np.sum(tau**2,axis=0) + u_star**2))))

    T = tau/new_gamma
    s = 1/(1 + np.sum(T**2,axis=0))
    # uprim,new_gamma,s,T = step_2(u_half)
    new_u = s*( uprim + (uprim[0]*T[0] + uprim[1]*T[1] + uprim[2]*T[2])*T + cross_axis0(uprim, T))
    new_VEL = new_u/new_gamma
    new_POS = POS + VEL*dt
    return new_POS, new_VEL, new_gamma

#---------------------------------------------------------------------------------

def average(X,Y,dr_av):
    M = []
    a_range = np.arange(0,np.max(X)*1.0,0.01)
    for a in a_range:
        inter = np.where((X > a-dr_av/2) & (X < a+dr_av/2))
        M.append(np.mean(Y[inter]))
    return a_range,np.array(M)

def save_conf(POS,GVEL):
    FOLDER = "Tp_w60.0_a1.0"
    print("==== SAVED FILES ====")
    print(f"{FOLDER}/POS_a{a0}_zR{zR/w0**2}_{f_str}_{g_str}_{eps}{l}_w{w0/l0:.1f}_Tp{Tp/l0:.0f}_dt{dt}_N{N}_Pz{PZ:.0f}_Lz{LZ:.0f}")
    print("=====================")
    file = f"FILE_PUSHER/{FOLDER}/POS_a{a0:.2f}_zR{zR/w0**2}_{f_str}_{g_str}_{eps}{l}_w{w0/l0:.1f}_Tp{Tp/l0:.0f}_dt{dt}_zfoc{zfoc/l0:.0f}_N{N}_Pz{PZ:.0f}_Lz{LZ:.0f}.npy"
    np.save(file, POS)
    file = f"FILE_PUSHER/{FOLDER}/GVEL_a{a0:.2f}_zR{zR/w0**2}_{f_str}_{g_str}_{eps}{l}_w{w0/l0:.1f}_Tp{Tp/l0:.0f}_dt{dt}_zfoc{zfoc/l0:.0f}_N{N}_Pz{PZ:.0f}_Lz{LZ:.0f}.npy"
    np.save(file, GVEL)

#==============================
# SPATIAL AND TEMPORAL ENVELOP
#=============================
pwr = 2
c_super=sqrt(2)*scipy.special.gamma((pwr+1)/pwr)

g = g_sin2
f = f_nuter
f_prime = f_prime_nuter

g_str = str(g).split(" ")[1]
f_str = str(f).split(" ")[1]
#==========================
# LASER PARAMETERS
#==========================
a0 = 2
w0 = 6*l0
Tp = 20*l0
zR = 0.5*w0**2

eps,l = 0,1
print(g_str,"| a0=",round(a0,2),f"| l={l},s={eps} | w0={w0/l0:.2f} | Tp={Tp/l0:.0f}")

C_lp = np.sqrt(1/factorial(abs(l)))*sqrt(2)**abs(l)
#==========================
# NUMERICAL PARAMETERS
#==========================
dt = 0.01
N = 1_000
zfoc = 0*l0
plasma_length = 0*l0
plasma_pos = 0*l0 + 1e-12
TMAX = plasma_pos + Tp*1.2 #9*l0
t_range = np.arange(0,TMAX,dt)

LONGITUDINAL_FIELD = True
SAVE = False

rmax = w0*2
if f == f_nuter: rmax=w0*2

# =========================
# NUTER INITIAL CONDITIONS
# =========================
theta0 = np.random.random(N)*2*pi
r0 = np.sqrt(np.random.uniform(0,1,N))*rmax
POS = np.zeros((3,N))
POS[2] = plasma_pos+np.random.random(N)*plasma_length
POS[0] = r0*cos(theta0)
POS[1] = r0*sin(theta0)
VEL = np.zeros((3,N))
gamma = np.ones(N)


# =========================
# INIT DIAGS
# =========================
POS_HIST = np.array([POS])
GVEL_HIST = np.array([gamma*VEL])
Lz_HIST = []
Pr_HIST = []
COMPUTE_TIME = []
t_range_plot = [0]
F_lorentz = []
Ex_HIST = []
Ez_HIST = []
# =========================
# PLOTS
# =========================
# point = plt.scatter(POS[0],POS[1],c=POS[0],s=1,cmap = "RdYlBu",vmin=-0.001,vmax=0.001)
# line, = plt.plot(POS_HIST[:,0],POS_HIST[:,1],"-")



for k,t in enumerate(tqdm(t_range)):
    t0 = time.perf_counter()
    x,y,z = POS
    tau = t - z
    r = sqrt(x**2+y**2)
    theta = arctan2(y,x)
    # t1 = time.perf_counter()
    # Ex,Ey,Ez = getE(x,y,z-zfoc,t,tau,r,theta)
    # Bx,By,Bz = getB(x,y,z-zfoc,t,tau,r,theta)
    Ex,Ey,Ez,Bx,By,Bz = get_EB_fast(x,y,z-zfoc,t)
    
    POS, VEL, gamma = RELAT_PUSH(np.array([Ex,Ey,Ez]), np.array([Bx,By,Bz]), dt, POS, VEL, gamma)
    # t3 = time.perf_counter()
    # print((t2-t1)*1000,"ms | ",(t3-t2)*1000,"ms")
    # POS, VEL = PUSH(E, B, dt, POS, VEL)

    # print(E[0])
    if k%int(0.2/dt)==0 or k==len(t_range)-1:
        # print(f"============= SAVE LAST TIME STEP t={t:.3f}=============")
        POS_HIST = np.vstack([POS_HIST,POS[None,:]])
        GVEL_HIST = np.vstack([GVEL_HIST,gamma*VEL[None,:]])
        Lz = gamma*(POS[0]*VEL[1] - POS[1]*VEL[0]) #POS = [x,y,z]
        Lz_HIST.append(Lz)
        pr = gamma*(POS[0]*VEL[0] + POS[1]*VEL[1])/sqrt(POS[0]**2+POS[1]**2)
        Pr_HIST.append(pr)
        t_range_plot.append(t)
        F_lorentz.append(np.array([Ex,Ey,Ez]) + 0*np.cross(VEL,np.array([Bx,By,Bz]),axis=0))
        Ex_HIST.append(Ex)
        Ez_HIST.append(Ez)



R_HIST = sqrt(POS_HIST[:,0]**2+POS_HIST[:,1]**2)
THETA_HIST = arctan2(POS_HIST[:,1],POS_HIST[:,0])
Lz_HIST = np.array(Lz_HIST)
Pr_HIST = np.array(Pr_HIST)
t_range_plot = np.array(t_range_plot)
F_lorentz = np.array(F_lorentz)
Ex_HIST = np.array(Ex_HIST)
Ez_HIST = np.array(Ez_HIST)

#=================================================
# MAIN 2X2 PLOT
#=================================================


y,z,x = POS_HIST[:,0], POS_HIST[:,1],POS_HIST[:,2]
r = sqrt(y**2+z**2)

py,pz,px = GVEL_HIST[:,0], GVEL_HIST[:,1],GVEL_HIST[:,2]


r0_requested = 1.7*l0

Nid = np.where(np.abs(r[0]-r0_requested)==np.min(np.abs(r[0]-r0_requested)))[0][0]


Ey, Ez, Ex, _, _, _ = get_EB_fast(y[:,Nid], z[:,Nid], x[:,Nid],t_range_plot)
Ay, Az, Ax, _, _, _ = get_EB_fast(y[:,Nid], z[:,Nid], x[:,Nid],t_range_plot+pi/2)

px_slow = savgol_filter(px[:,Nid], window_length=201,polyorder=2)


plt.figure()
plt.plot(t_range_plot/l0, Ax, label="Ax")
# plt.plot(t_range_plot/l0, px[:,Nid], label="px")
plt.plot(t_range_plot/l0, px[:,Nid]-px_slow, "--",label="px_f")

plt.grid()
plt.xlabel("t/t0")
plt.legend()
# plt.plot(px[:,Nid])










