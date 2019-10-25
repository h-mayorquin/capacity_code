#!/bin/bash -l

#SBATCH -A 2018-3-582
#SBATCH -t 04:00:00
#SBATCH -J python_work

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

module load mpi4py/3.0.2/py37
aprun -n 128 -N 32 python mpi_test.py > my_output_file.txt
