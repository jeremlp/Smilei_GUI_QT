import sys
import subprocess
import time
import os
import datetime
import json


def getSimProgress(path):

    cmd_cat = os.popen(f"cat {path}/log").read()

    time_info_str = cmd_cat.split("\n")[-2].split()

    progress_list = time_info_str[0].split("/")
    try:
        current_timestep = int(progress_list[0])
        total_timestep = int(progress_list[1])

        timestep_every = float(cmd_cat.split("\n")[-2].split()[0].split("/")[0]) - float(cmd_cat.split("\n")[-3].split()[0].split("/")[0])
        progress = current_timestep/total_timestep
        last_compute_time = float(time_info_str[4]) #nb of second of last time step
        ETA_minutes= (total_timestep - current_timestep)/timestep_every*last_compute_time/60
        ETA_str = f'{ETA_minutes//60:.0f}h{ETA_minutes%60:0>2.0f}'
    except:
        progress = 0
        ETA_str = "-"
    return progress, ETA_str

def writeToFile(sim_info_dict_main):
    with open('/sps3/jeremy/LULI/simulation_info.json', 'w') as file:
        json.dump(sim_info_dict_main, file, indent=4)
    return 1


TOTAL_JOB_COUNT = 0
k = 0
sim_info_dict_main = {}

cmd = os.popen("PBS_DEFAULT=poltrnd.in2p3.fr qstat -u jeremy").read()
res_list = cmd.split("\n")[5:-1]

for i,sim in enumerate(res_list):
    sim_list = sim.split()
    job_id = sim_list[0].split(".")[0]
    sim_runtime = sim_list[-1]
    nb_nodes = sim_list[5]

    cmd_full = os.popen(f"PBS_DEFAULT=poltrnd.in2p3.fr qstat -f {job_id}").read()

    res_full_list = cmd_full.split("=")

    for n,value in enumerate(res_full_list):
        if "PBS_O_WORKDIR" in value:
            n_path = n+1
            break
    
    str_path = res_full_list[n_path].split(" ")[0]
    job_full_path = repr(str_path).replace("\\n","").replace("\\t","").strip("'")

    job_full_name = res_full_list[1].split("\n")[0][1:]
    

    progress, ETA_str = getSimProgress(job_full_path)
    
    sim_info_dict = {'job_id':job_id, 
                'job_full_path':job_full_path, 
                'job_full_name':job_full_name,
                'NODES': nb_nodes, 
                'progress':progress, 
                'ETA':ETA_str}
    sim_info_dict_main[job_id] = sim_info_dict
    
now = datetime.datetime.now()
datetime_str = now.strftime("%d/%m - %Hh%M")
sim_info_dict_main["datetime"] = datetime_str
print(sim_info_dict_main)
a = writeToFile(sim_info_dict_main)
