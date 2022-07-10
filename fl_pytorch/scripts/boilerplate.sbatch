#!/bin/bash -l

#SBATCH --job-name=mpi4py-test   # create a name for your job
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --ntasks-per-node=1      # number of tasks per node (totally 100 clients and one master)
#SBATCH --gres=gpu:1             # number of gpus per node

#SBATCH --time=312:00:00                   # total run time limit (HH:MM:SS)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=my_mail@kaust.edu.sa   # your email

#SBATCH --mem=100G              # RAM per node
#SBATCH --threads-per-core=1    # do not use hyperthreads (i.e. CPUs = physical cores below)
#SBATCH --cpus-per-task=4       # number of CPUs per process

conda activate fl
