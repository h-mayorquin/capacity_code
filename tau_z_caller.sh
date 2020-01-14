#!/bin/bash -l

#SBATCH -A 2019-3-58
#SBATCH -t 01:00:00
#SBATCH -J tau_z

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

#SBATCH -e error_file.e
#SBATCH -o printed.o

hypercolumns=10
minicolumns=10
sequence_length=4
transitions=50
sigma=0
tau_a=0.150
g_a=4.0
T_start=0.100
T_per_sequence=0.100
recall_dynamics="normalma"
#recall_dynamics="normalno" 

output="../databases/tau_z_h${hypercolumns}_m${minicolumns}_sl${sequence_length}_transitions${transitions}_sigma${sigma}_a${tau_a}_ga${g_a}_${recall_dynamics}.pickle"

module load mpi4py/3.0.2/py37
#aprun -n 128 -N 32 python ./mpi_test.py > my_output_file.txt
srun -n 128 python ./sequence_curve.py $hypercolumns $minicolumns $sequence_length $transitions $sigma $tau_a $g_a $T_start $T_per_sequence $recall_dynamics $output
