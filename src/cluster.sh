#!/usr/bin/env bash
#SBATCH --mincpus 4
#SBATCH --mem 4000
#SBATCH --exclude rockford,steele,hammer,conan,blomquist,wolfe,knatterton,holmes,lenssen,scuderi,matula,marlowe,poirot,monk

##SBATCH --nodelist=turbine


source $HOME/.software/python_environments/tensorflow_v2/bin/activate
srun -u python3 gym_experiment.py "$@"
