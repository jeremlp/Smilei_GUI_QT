# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:08:28 2024

@author: jerem
"""
import sys
import pexpect
from pexpect.popen_spawn import PopenSpawn

p = PopenSpawn('cmd.exe', encoding='utf-8',codec_errors='ignore')
p.logfile_read   = sys.stdout

# fout = open('mylog.txt','wb')
# child.logfile_send = fout

#r = subprocess.call(["scp", f"jeremy@llrlsi-gw.in2p3.fr:\sps3\jeremy\LULI\{sim_json_name}","."])

r = p.sendline("scp jeremy@llrlsi-gw.in2p3.fr:\sps3\jeremy\LULI\simulation_info.json .")
# index = p.sendline('dir')
p.expect(pexpect.EOF)
# print(p.before.rstrip())

# print(index)
# child.wait()
# index = child.expect('password:')
# print(index)
