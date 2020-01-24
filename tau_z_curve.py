from mpi4py import MPI
import warnings
import pickle 
import itertools
import sys

import numpy as np
import random
from math import ceil, floor
from copy import deepcopy
warnings.filterwarnings('ignore')

from functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Run a dummy example

hypercolumns = int(sys.argv[1])
minicolumns = int(sys.argv[2])
sequence_length = int(sys.argv[3])
transitions = int(sys.argv[4])
sigma = float(sys.argv[5])
tau_a = float(sys.argv[6])
g_a = float(sys.argv[7])
T_start=float(sys.argv[8])
T_per_pattern=float(sys.argv[9])
recall_dynamics = sys.argv[10]

memory = recall_dynamics[-2]

tau_z_slow = 0.005


total_trials = 128 * 2

trials_per_rank = ceil(total_trials / size)
total_trials = trials_per_rank * size
transitions_per_sequence = sequence_length - 1
ns = floor(transitions / transitions_per_sequence)
tau_z_vector = np.linspace(0.010, 0.125, num=25)


if rank == 0:
    storage_dic_success = {}
    storage_dic_points_of_failure = {}
    storage_dic_persistent_times = {}
    storage_dic_pairs = {}
    
pattern_seed = rank
for tau_z_pre in tau_z_vector:
    
    aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, ns, sequence_length, pattern_seed, tau_z_pre, 
                         sigma, tau_z_slow, tau_a, g_a, memory, recall_dynamics, T_start, T_per_pattern)
    successes, points_of_failure, persistence_times, seq_recalled_pairs = aux
    
    aux_success = comm.gather(successes, root=0)
    aux_failure = comm.gather(points_of_failure, root=0)
    aux_times = comm.gather(persistence_times, root=0)
    aux_pairs = comm.gather(seq_recalled_pairs, root=0)
    
    if rank == 0:
        storage_dic_success[tau_z_pre] = sum(aux_success, [])
        storage_dic_points_of_failure[tau_z_pre] = sum(aux_failure, [])
        storage_dic_persistent_times[tau_z_pre] = sum(aux_times, [])
        storage_dic_pairs[tau_z_pre] = sum(aux_pairs, [])
        
# Store data as a pickle
if rank == 0:
    save_dic = {'success': storage_dic_success, 'points_of_failure':storage_dic_points_of_failure, 'persistent_times':storage_dic_persistent_times, 
               'hypercolumns': hypercolumns, 'minicolumns': minicolumns, 'number_of_sequences':ns, 'sigma':sigma, 'g_a':g_a,
                'sequence_length': sequence_length, 'trials':total_trials, 'tau_z_pre_vector':tau_z_vector, 
                'tau_a':tau_a, 'memory':memory, 'recall_dynamics':recall_dynamics}
    
    filename = sys.argv[11]
    with open(filename, 'wb') as handle:
        pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)