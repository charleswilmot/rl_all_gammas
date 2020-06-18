import omegaconf
import hydra
import os
from paramiko import SSHClient, RSAKey, AutoAddPolicy
from getpass import getpass


PASSWORD = None

@hydra.main(config_path='../config/prototype_config.yml')
def run_on_cluster(cfg):
    command_line_args = serialize_cfg(cfg)
    command_line_args += additional_args()
    job_name = get_job_name()
    # print(command_line_args)
    # print(job_name)
    output_flag = "--output {outdir}/%N_%j.log".format(outdir=os.getcwd())
    job_name_flag = "--job-name {job_name}".format(job_name=job_name)
    command_line = "sbatch {output_flag} {job_name_flag} cluster.sh\\\n".format(
        output_flag=output_flag,
        job_name_flag=job_name_flag
    ) + command_line_args
    print("###")
    print(command_line)
    print("###")
    start_job_over_ssh(command_line)


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


def start_job_over_ssh(sbatch_cmd):
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
    stdin, stdout, stderr = client.exec_command('(cd Documents/code/rl_all_gammas/src ; {})'.format(sbatch_cmd))
    for line in stdout.readlines():
        print(line)


if __name__ == "__main__":
    run_on_cluster()
