# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:12:44 2024

@author: jerem
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi
plt.close("all")
S = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_PHYSICAL/sim_base_OAM_Long')
S2 = happi.Open(f'{os.environ["SMILEI_CLUSTER"]}/SIM_PHYSICAL/sim_base_SAM_Long')

l0=2*np.pi

weight_diag = S.ParticleBinning("weight_av")
weight = np.mean(np.array(weight_diag.getData())[-1,:,:]/0.03,axis=-1)
weight_av = np.mean(weight, axis=0)
weight_diag2 = S2.ParticleBinning("weight_av")
weight2 = np.mean(np.array(weight_diag2.getData())[-1,:,:]/0.03,axis=-1)
weight_av2 = np.mean(weight2, axis=0)

x_range = weight_diag.getAxis("x")/l0
r_range = weight_diag.getAxis("y")/l0- S.namelist.Ltrans/2/l0

fig, (ax1,ax2) = plt.subplots(2,figsize=(12,7))
im = ax1.imshow(weight.T, cmap="jet",aspect="auto",vmin=1,vmax=1.5, extent=[x_range[0], x_range[-1], r_range[0],r_range[-1]])
plt.colorbar(im,ax=ax1)
im2 = ax2.imshow(weight2.T, cmap="jet",aspect="auto",vmin=1,vmax=1.5, extent=[x_range[0], x_range[-1], r_range[0],r_range[-1]])
plt.colorbar(im2,ax=ax2)
fig.tight_layout()


l_list = []
x_list = []
r1_list = []
r2_list = []

r1_list2 = []
r2_list2 = []

l_list2 = []
x_list2 = []
for x_idx in range(len(x_range)):
    peaks, _ = find_peaks(weight[x_idx], height=1.05, prominence=0.001)
    peaks = peaks[(np.abs(r_range[peaks])<9) &(np.abs(r_range[peaks])>5)]
    peaks2, _ = find_peaks(weight2[x_idx], height=1.5, prominence=0.35)

    np.where(weight[x_idx]<1.0)

    # print(x_range[x_idx],peaks)
    if len(peaks) >=2:

        l_channel = (r_range[peaks[-1]] - r_range[peaks[0]])/2
        if l_channel >1.5:
            l_list.append(l_channel)
            x_list.append(x_range[x_idx])
            r1_list.append(r_range[peaks[0]])
            r2_list.append(r_range[peaks[-1]])
            ax1.scatter(x_range[x_idx],r_range[peaks[0]],marker="x",color="green")
            ax1.scatter(x_range[x_idx],r_range[peaks[-1]],marker="x",color="green")

    print(peaks2)
    if len(peaks2)==2:
        l_channel2 = (r_range[peaks2[-1]] - r_range[peaks2[0]])/2
        if l_channel2 <10 and l_channel2 >4:

            l_list2.append(l_channel2)
            x_list2.append(x_range[x_idx])
            r1_list2.append(r_range[peaks2[0]])
            r2_list2.append(r_range[peaks2[-1]])
            ax2.scatter(x_range[x_idx],r_range[peaks2[0]],marker="x",color="red")
            ax2.scatter(x_range[x_idx],r_range[peaks2[-1]],marker="x",color="red")


# plt.figure()

# plt.plot(r_range, weight_av,label="OAM")
# plt.plot(r_range, weight_av2, label="SAM")

# plt.grid()
# plt.xlabel("$r/\lambda$")
# plt.ylabel("Density ne/nc")
plt.figure()
plt.plot(x_list,l_list, label="OAM")
plt.plot(x_list2,l_list2, label="SAM")
plt.grid()
plt.xlabel("$x/\lambda$")
plt.ylabel("Channel size $l/\lambda$")
plt.legend()
plt.title("Channel size using density peaks")


azeeazeaz
l_list_hole_1 = []
l_list_hole_2 = []
for x_idx in range(len(x_range)):
    below_one = np.diff(np.where(weight[x_idx] < 1.0)[0])
    diff = np.diff(below_one)
    splits = np.where(diff > 1)[0] + 1
    holes = np.split(below_one, splits)
    hole_lengths = [len(hole) for hole in holes]
    dr = r_range[1] - r_range[0]
    l_list_hole_1.append(dr*np.max(hole_lengths))

    below_one = np.diff(np.where(weight2[x_idx] < 1.0)[0])
    diff = np.diff(below_one)
    splits = np.where(diff > 1)[0] + 1
    holes = np.split(below_one, splits)
    hole_lengths = [len(hole) for hole in holes]
    dr = r_range[1] - r_range[0]
    l_list_hole_2.append(dr*np.max(hole_lengths)/2)


plt.figure()
plt.plot(x_range,l_list_hole_1,label="OAM")
plt.plot(x_range,l_list_hole_2,label="SAM")
plt.ylim(0,9)
plt.grid()
plt.legend()
plt.title("Channel size using density bellow 1.0")

