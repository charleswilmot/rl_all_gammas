#!/usr/bin/env sh
#SBATCH --mincpus 4
#SBATCH --mem 4000


source ~/.software/python_environments/tensorflow_v2/bin/activate
srun -u python3 gym_experiment.py "$@"
