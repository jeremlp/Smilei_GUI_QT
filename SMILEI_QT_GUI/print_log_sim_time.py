# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:30:39 2024

@author: jerem
"""

import os
import numpy as np

folder = "D:\JLP\CMI\_MASTER_2_\_STAGE_LULI_\CLUSTER\performance"
# folder = "D:\JLP\CMI\_MASTER_2_\_STAGE_LULI_\CLUSTER"
subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
# subfolders = [r"D:\JLP\CMI\_MASTER_2_\_STAGE_LULI_\CLUSTER\test_base_script_plasma_OAM_XDIR_NO_IONS",
#               r"D:\JLP\CMI\_MASTER_2_\_STAGE_LULI_\CLUSTER\test_base_script_plasma_OAM_XDIR"]
for sim_path in subfolders:
    if sim_path.split("\\")[-1] == "performance": continue
    path_log = sim_path+"\\log"
    with open(path_log) as f:
        text = f.readlines()
    isDiag = False
    isFinished = False

    print("\n=============================")
    print(sim_path.split("\\")[-1])
    print("=============================")
    for i, line in enumerate(text):
        if "push time [ns]" in line:
            pt = int(np.mean([int(text[i+n].split()[-1]) for n in range(1,40)]))
            print("Push time:",pt,"ns")
            print("-----------------")
        if "Time_in_time_loop" in line and not isFinished:
            print("TOTAL:",round(float(line.split()[1])/60),"min")
            isDiag = True
        if "Particles" in line and isDiag and not isFinished:
            print("Particles:",round(float(line.split("\t")[2])/60),"min")
        if "Maxwell_BC" in line and isDiag and not isFinished:
            print("Maxwell_BC:",round(float(line.split("\t")[2])/60),"min")
        if "Diagnostics" in line and isDiag and not isFinished:
            print("Diagnostics:",round(float(line.split("\t")[2])/60),"min")
            isFinished = True