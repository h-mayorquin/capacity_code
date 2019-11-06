#!/bin/bash -l

#SBATCH -A 2018-3-582
#SBATCH -t 23:00:00
#SBATCH -J hypercolumns_run

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

#SBATCH -e error_file.e
#SBATCH -o output_file.o

module load mpi4py/3.0.2/py37
#aprun -n 128 -N 32 python ./mpi_test.py > my_output_file.txt
srun -n 1 python ./calculate_root_hypercolumns.py > ./output_files/hypercolumns_run.txt

