#!/bin/bash -l

#SBATCH -A 2018-3-582
#SBATCH -t 01:00:00
#SBATCH -J sequence

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32

#SBATCH -e error_file.e
#SBATCH -o printed.o


hypercolumns=10
minicolumns=10
sequence_length=4
n_transitions=40
tau_z_pre=0.050
tau_a=0.150
trials=400
recall_dynamics="normala"

hypercolumns = int(sys.argv[1])
minicolumns = int(sys.argv[2])
sequence_length = int(sys.argv[3])
number_of_transitions = sys.argv[4]
tau_z_pre = float(sys.argv[5])
tau_a = float(sys.argv[6])
recall_dynamics = sys.argv[7]

output="./point_data_h${hypercolumns}_m${minicolumns}_sl${sequence_length}_nt${n_transitions}_z${tau_z_pre}_${recall_dynamics}.pickle"

module load mpi4py/3.0.2/py37
#aprun -n 128 -N 32 python ./mpi_test.py > my_output_file.txt
srun -n 128 python ./sequence_curve.py $hypercolumns $minicolumns $sequence_length $n_transitions $tau_z_pre $tau_a $recall_dynamics $trials $output
