#!/bin/bash -l

#SBATCH -A 2019-3-58
#SBATCH -t 00:30:00
#SBATCH -J g_I

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

#SBATCH -e error_file.e
#SBATCH -o printed.o

hypercolumns=10
minicolumns=10
sequence_length=4
transitions=40
tau_z_pre=0.025
sigma=0
tau_a=0.150
g_a=4.0
T_start=0.100
T_per_sequence=0.100
recall_dynamics="normalma"
#recall_dynamics="normalno"
remove=0.010

output="../databases/cue_h${hypercolumns}_m${minicolumns}_sl${sequence_length}_transitions${transitions}_z${tau_z_pre}_sigma${sigma}_a${tau_a}_ga${g_a}_Tmin${remove}_${recall_dynamics}.pickle"

module load mpi4py/3.0.2/py37
#aprun -n 128 -N 32 python ./mpi_test.py > my_output_file.txt
srun -n 128 python ./cue_curve.py $hypercolumns $minicolumns $sequence_length $transitions $tau_z_pre $sigma $tau_a $g_a $T_start $T_per_sequence $remove $recall_dynamics $output
