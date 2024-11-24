# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:35:27 2024

@author: jerem
"""

import hashlib

def generate_diag_id(file, insert_id=False):
    print("generate_diag_id:",file)

    with open(file) as f:
        lines = f.readlines()
    cond=0
    diag_id_idx = None
    diag_idx = None
    for i,l in enumerate(lines):
        if "DIAG_ID:" in l:
            diag_id_idx = i
        if l == "#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#\n":
            cond +=1
            if cond ==2:
                diag_idx = i
                break
    if diag_idx is None: #Log has not been loaded
        return -1, -1
    diag_txt = '\n'.join(lines[diag_idx:])
    import hashlib
    hashing_func = hashlib.md5
    str2int = lambda s : int(hashing_func(s.encode()).hexdigest(), 16)

    diag_id_full = str2int(diag_txt)
    diag_id = int(str(diag_id_full)[::9])

    if insert_id: #insert diag_id in namelist
        if diag_id_idx is None:
            lines.insert(diag_idx-5, f"\n#DIAG_ID: {diag_id}")
        else:
            lines[diag_id_idx] = f"#DIAG_ID: {diag_id}\n"

        with open(file, "w") as f:
            f.writelines(lines)
        print(f"Inserted '#DIAG_ID: {diag_id}' in namelist")

    return diag_id_full, diag_id


def get_diag_id(file):
        with open(file) as f:
            lines = f.readlines()
        for i,l in enumerate(lines):
            if "DIAG_ID:" in l:
                diag_id = l.split(" ")[-1]
                return int(diag_id)
        diag_id_full, diag_id = generate_diag_id(file)
        return int(diag_id)


if __name__ == '__main__':
    import sys
    import os
    file = f'{os.environ["SMILEI_CLUSTER"]}/SIM_PHYSICAL/sim_OAM_Long/laser_propagation_3d.py'
    print(generate_diag_id(file,insert_id=True))
    print(get_diag_id(file))



