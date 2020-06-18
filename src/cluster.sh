#!/usr/bin/env sh
#SBATCH --mincpus 4
#SBATCH --mem 4000


source $HOME/.software/python_environments/tensorflow_v2/bin/activate
echo "test"
pip3 list
srun -u python3 gym_experiment.py "$@"
