# -*- coding: utf-8 -*-

"""
FROM https://hackersandslackers.com/automate-ssh-scp-python-paramiko/
"""

"""Client to handle connections and actions executed against a remote host."""
from os import system
from typing import List

from paramiko import AutoAddPolicy, RSAKey, SSHClient
from paramiko.auth_handler import AuthenticationException, SSHException
from scp import SCPClient, SCPException

# from logging import LOGGER


class RemoteClient:
    """Client to interact with a remote host via SSH & SCP."""

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        ssh_key_filepath: str,
        remote_path: str,
    ):
        self.host = host
        self.user = user
        self.password = password
        self.ssh_key_filepath = ssh_key_filepath
        self.remote_path = remote_path
        self.client = None
        #self._upload_ssh_key()

    @property
    def connection(self):
        """Open SSH connection to remote host."""
        try:
            client = SSHClient()
            print("SSH")
            client.load_system_host_keys()
            client.set_missing_host_key_policy(AutoAddPolicy())
            client.connect(
                self.host,
                username=self.user,
                password=self.password,
                key_filename=self.ssh_key_filepath,
                timeout=5000,
            )
            return client
        except AuthenticationException as e:
            print(
                f"AuthenticationException occurred; did you remember to generate an SSH key? {e}"
            )
        except Exception as e:
            print(f"Unexpected error occurred while connecting to host: {e}")

    @property
    def scp(self) -> SCPClient:
        conn = self.connection
        return SCPClient(conn.get_transport())

    def _get_ssh_key(self):
        """Fetch locally stored SSH key."""
        try:
            self.ssh_key = RSAKey.from_private_key_file(self.ssh_key_filepath)
            print(f"Found SSH key at self {self.ssh_key_filepath}")
            return self.ssh_key
        except SSHException as e:
            print(f"SSHException while getting SSH key: {e}")
        except Exception as e:
            print(f"Unexpected error while getting SSH key: {e}")

    def _upload_ssh_key(self):
        try:
            system(
                f"ssh-copy-id -i {self.ssh_key_filepath}.pub {self.user}@{self.host}>/dev/null 2>&1"
            )
            print(f"{self.ssh_key_filepath} uploaded to {self.host}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError while uploading SSH key: {e}")
        except Exception as e:
            print(f"Unexpected error while uploading SSH key: {e}")

    def disconnect(self):
        """Close SSH & SCP connection."""
        if self.connection:
            self.client.close()
        if self.scp:
            self.scp.close()

    def bulk_upload(self, filepaths: List[str]):
        """
        Upload multiple files to a remote directory.

        :param List[str] filepaths: List of local files to be uploaded.
        """
        try:
            self.scp.put(filepaths, remote_path=self.remote_path, recursive=True)
            print(
                f"Finished uploading {len(filepaths)} files to {self.remote_path} on {self.host}"
            )
        except SCPException as e:
            print(f"SCPException during bulk upload: {e}")
        except Exception as e:
            print(f"Unexpected exception during bulk upload: {e}")

    def download_file(self, filepath: str, local_folder : str):
        """
        Download file from remote host.

        :param str filepath: Path to file hosted on remote server to fetch.
        """
        self.scp.get(filepath, local_folder)

    def bulk_download(self, filepaths: List[str], local_folder : str):
        try:
            self.scp.get(filepaths, local_folder, recursive=True)
            print(
                f"Finished downloading {len(filepaths)} files from {self.remote_path}"
            )
        except SCPException as e:
            print(f"SCPException during bulk download: {e}")
        except Exception as e:
            print(f"Unexpected exception during bulk download: {e}")

    def execute_commands(self, commands: List[str]):
        """
        Execute multiple commands in succession.

        :param List[str] commands: List of unix commands as strings.
        """
        for cmd in commands:
            stdin, stdout, stderr = self.connection.exec_command(cmd)
            stdout.channel.recv_exit_status()
            response = stdout.readlines()
            # error = stderr.readlines()
            # print("SSH IN:",stdin.readlines())
            # print("SSH ERROR:",stderr.readlines())
            # print("SSH OUT:",stdout.readlines())
            if len(stderr.readlines())>0:
                print("SSH ERROR:",stderr.readlines())
            for line in response:
                print(
                    f"INPUT: {cmd}\n \
                    OUTPUT: {line}"
                )
        return response



if __name__ == '__main__':
    from utils import Popup, encrypt
    import os
    host = "llrlsi-gw.in2p3.fr"
    user = "jeremy"
    with open(f"{os.environ['SMILEI_QT']}\\..\\tornado_pwdfile.txt",'r') as f: pwd_crypt = f.read()
    pwd = encrypt(pwd_crypt,-2041000*2-1)
    remote_path = r"\sps3\jeremy\LULI\simulations_info.json"
    ssh_key_filepath = r"C:\Users\Jeremy\.ssh\id_rsa.pub"
    remote_client = RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
    remote_client.execute_commands(["python3 /sps3/jeremy/LULI/check_sim_state_py.py"])




