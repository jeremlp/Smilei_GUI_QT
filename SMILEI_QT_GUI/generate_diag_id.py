# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:35:27 2024

@author: jerem
"""

import hashlib

def get_diag_id(file):

    with open(file) as f:
        lines = f.readlines()
    cond=0
    for i,l in enumerate(lines):
        if l == "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#\n":
            cond +=1
            if cond ==2:
                diag_idx = i
                break
    diag_txt = '\n'.join(lines[diag_idx:])
    import hashlib
    hashing_func = hashlib.md5
    str2int = lambda s : int(hashing_func(s.encode()).hexdigest(), 16)

    v = str2int(diag_txt)
    return v,int(str(v)[::9])

if __name__ == '__main__':
    import sys
    file = 'C:/_DOSSIERS_PC/_STAGE_LULI_/CLUSTER/SIM_PHYSICAL/sim_OAM_Long/laser_propagation_3d.py'
    print(get_diag_id(file))

