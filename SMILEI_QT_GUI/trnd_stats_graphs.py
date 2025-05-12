# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 09:26:51 2025

@author: Jeremy
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils import Popup, encrypt
import paramiko_SSH_SCP_class


MAX_NODES_CLAKE = 28
MAX_NODES_SKYLAKE = 16
MAX_NODES_DEFAULT = 8

base_path = fr'{os.environ["SMILEI_QT"]}/'


host = "llrlsi-gw.in2p3.fr"
user = "jeremy"
with open(f"{os.environ['SMILEI_QT']}\\..\\..\\tornado_pwdfile.txt",'r') as f: pwd_crypt = f.read()
pwd = encrypt(pwd_crypt,-2041000*2-1)
remote_path = r"\sps3\jeremy\LULI\simulations_info.json"
ssh_key_filepath = r"C:\Users\Jeremy\.ssh\id_rsa.pub"
remote_client = paramiko_SSH_SCP_class.RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
remote_client.execute_commands(["(source /usr/share/Modules/init/bash; unset MODULEPATH; module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7; module load python/3.7.0; python /sps3/jeremy/LULI/check_sim_state_py.py)&"])
remote_client.download_file(r"/sps3/jeremy/LULI/cluster_user_infos.txt", rf"{os.environ['SMILEI_QT']}")

file = base_path + "cluster_user_infos.txt"


data = np.loadtxt(file,dtype=str)

df = pd.DataFrame(data, columns=['date', 'time', 'id', 'user', 'partition', 'nodes'])

df['nodes'] = df['nodes'].astype(int)
df['id'] = df['id'].astype(int)

df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['time_bin'] = df['datetime'].dt.floor('1H')

df['time_bin_str'] = df['time_bin'].dt.strftime('%d %H:%M')

grouped = df.groupby(['time_bin_str', 'user'])['nodes'].max().unstack(fill_value=0)

grouped.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab10')

plt.title('Nodes by User')
plt.xlabel('Time')
plt.ylabel('Total Nodes used')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Users')
plt.grid(axis="y")
plt.tight_layout()



