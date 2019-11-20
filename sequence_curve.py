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
tau_z_pre = float(sys.argv[4])
tau_a = float(sys.argv[5])
hypercolumns = int(sys.argv[2])
minicolumns = int(sys.argv[3])
sequence_length = int(sys.argv[1])
number_of_sequences = 3
total_trials = 100
trials_per_rank = ceil(total_trials / size)
max_transitions = 100
transitions_per_sequence = sequence_length - 1
max_ns = floor(max_transitions / transitions_per_sequence)
number_of_sequences_vector = np.arange(2, max_ns , 1)

if rank == 0:
    storage_dic_success = {}
    storage_dic_points_of_failure = {}
    storage_dic_persistent_times = {}
    storage_dic_pairs = {}
    
pattern_seed = rank
for ns in number_of_sequences_vector:
    
    aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, ns, sequence_length, pattern_seed, tau_z_pre, tau_a)
    successes, points_of_failure, persistence_times, seq_recalled_pairs = aux
    
    aux_success = comm.gather(successes, root=0)
    aux_failure = comm.gather(points_of_failure, root=0)
    aux_times = comm.gather(persistence_times, root=0)
    aux_pairs = comm.gather(seq_recalled_pairs, root=0)
    
    if rank == 0:
        storage_dic_success[ns] = sum(aux_success, [])
        storage_dic_points_of_failure[ns] = sum(aux_failure, [])
        storage_dic_persistent_times[ns] = sum(aux_times, [])
        storage_dic_pairs[ns] = sum(aux_pairs, [])
        
# Store data as a pickle
if rank == 0:
    save_dic = {'success': storage_dic_success, 'points_of_failure':storage_dic_points_of_failure, 'persistent_times':storage_dic_persistent_times, 
               'hypercolumns': hypercolumns, 'minicolumns': minicolumns, 'number_of_sequences':number_of_sequences_vector, 
                'sequence_length': sequence_length, 'pairs':storage_dic_pairs, 'trials':total_trials, 'tau_z_pre':tau_z_pre, 'tau_a':tau_a}
    
    filename = sys.argv[6]
    with open(filename, 'wb') as handle:
        pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)