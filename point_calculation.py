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
number_of_transitions = int(sys.argv[4])
tau_z_pre = float(sys.argv[5])
tau_a = float(sys.argv[6])
recall_dynamics = sys.argv[7]
total_trials = int(sys.argv[8])

tau_z_slow = 0.005
memory=True
trials_per_rank = ceil(total_trials / size)
transitions_per_sequence = sequence_length - 1
number_of_sequences = ceil(number_of_transitions / transitions_per_sequence)


pattern_seed = rank
    
aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed, tau_z_pre, tau_z_slow, tau_a, memory, recall_dynamics)
successes, points_of_failure, persistence_times, seq_recalled_pairs = aux
    
aux_success = comm.gather(successes, root=0)
aux_failure = comm.gather(points_of_failure, root=0)
aux_times = comm.gather(persistence_times, root=0)
aux_pairs = comm.gather(seq_recalled_pairs, root=0)
    
if rank == 0:
    storage_success = sum(aux_success, [])
    storage_points_of_failure = sum(aux_failure, [])
    storage_persistent_times = sum(aux_times, [])
    storage_pairs = sum(aux_pairs, [])
        
# Store data as a pickle
if rank == 0:
    save_dic = {'success': storage_success, 'points_of_failure':storage_points_of_failure, 'persistent_times':storage_persistent_times, 
               'hypercolumns': hypercolumns, 'minicolumns': minicolumns, 'number_of_sequences':number_of_sequences, 
                'sequence_length': sequence_length, 'trials':total_trials, 'tau_z_pre':tau_z_pre, 'tau_a':tau_a, 'memory':memory, 'recall_dynamics':recall_dynamics}
    
    filename = sys.argv[9]
    with open(filename, 'wb') as handle:
        pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)