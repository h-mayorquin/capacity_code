#!/bin/bash -l

#SBATCH -A 2018-3-582
#SBATCH -t 01:00:00
#SBATCH -J sequence

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

#SBATCH -e error_file.e
#SBATCH -o printed.o

sequence_length=4
hypercolumns=10
minicolumns=10
tau_z_pre=0.050
tau_a=0.150
recall_dynamics="normala"

output="./data_h${hypercolumns}_m${minicolumns}_sl${sequence_length}_z${tau_z_pre}_${recall_dynamics}.pickle"

module load mpi4py/3.0.2/py37
#aprun -n 128 -N 32 python ./mpi_test.py > my_output_file.txt
srun -n 128 python ./sequence_curve.py $sequence_length $hypercolumns $minicolumns $tau_z_pre $tau_a $recall_dynamics $output
