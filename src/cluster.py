import omegaconf
import hydra
import os
from hydra.utils import get_original_cwd
from paramiko import SSHClient, RSAKey, AutoAddPolicy
from getpass import getpass
from socket import gethostname
import sys


PASSWORD = None
REMOTE_HOST_NAME = 'otto'


@hydra.main(config_path='../config/config.yaml', strict=True)
def send_job(cfg):
        command_line_args = serialize_cfg(cfg)
        command_line_args += additional_args()
        job_name = get_job_name()
        # print(command_line_args)
        # print(job_name)
        output_flag = "--output {outdir}/%N_%j.joblog".format(outdir=os.getcwd())
        job_name_flag = "--job-name {job_name}".format(job_name=job_name)
        partition_flag = "--partition {partition}".format(partition="sleuths")
        os.chdir(get_original_cwd())
        command_line = "sbatch {output_flag} {job_name_flag} {partition_flag} cluster.sh\\\n".format(
            output_flag=output_flag,
            job_name_flag=job_name_flag,
            partition_flag=partition_flag,
        ) + command_line_args
        os.system(command_line)


def additional_args():
    return " hydra.run.dir=" + os.getcwd() + "\\\n"


def get_job_name():
    return os.path.basename(os.getcwd())


def serialize_cfg(cfg, prefix=[]):
    if isinstance(cfg, omegaconf.dictconfig.DictConfig):
        string = ""
        for key in cfg:
            string += serialize_cfg(cfg[key], prefix=prefix + [key])
        return string
    else:
        return " {}={}\\\n".format(".".join(prefix), str(cfg))


def ssh_command(cmd):
    global PASSWORD
    host="fias.uni-frankfurt.de"
    user="wilmot"
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.load_system_host_keys()
    if PASSWORD is None:
        PASSWORD = getpass("Please enter password for the rsa key .ssh/id_rsa\n")
    pkey = RSAKey.from_private_key_file("/home/cwilmot/.ssh/id_rsa", password=PASSWORD)
    client.connect(host, username=user, pkey=pkey)
    stdin, stdout, stderr = client.exec_command('(cd Documents/code/rl_all_gammas/src ; source $HOME/.software/python_environments/tensorflow_v2/bin/activate ; {})'.format(cmd))
    for line in stdout.readlines():
        print(line, end='')
    for line in stderr.readlines():
        print(line, end='')
    print("")


if __name__ == "__main__" and gethostname() == REMOTE_HOST_NAME:
    send_job()
if __name__ == "__main__" and gethostname() != REMOTE_HOST_NAME:
    ssh_command("python3 " + " ".join(sys.argv))
