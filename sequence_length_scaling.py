from mpi4py import MPI
import warnings
import pickle 

warnings.filterwarnings('ignore')

import numpy as np
import random
from math import ceil, floor
from copy import deepcopy
from functions import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Run a dummy example
hypercolumns = 5
minicolumns = 10
total_trials = 200
trials_per_rank = ceil(total_trials / size)
trials = trials_per_rank * size


sequence_lengths_vector = [3, 4, 5, 6, 7, 8, 9]
number_of_sequences_vector = [2, 3, 4, 5, 6, 7, 8]
pattern_seed = rank

if rank == 0:
    p_small_dic = {}
    p2_dic ={ }
    p_diff_dic = {}
    
for sl, ns in itertools.product(sequence_lengths_vector, number_of_sequences_vector):
    # Large sequence
    sequence_length_large = sl
    number_of_sequences = ns
    sequence_length = sequence_length_large
    n_transitions = number_of_sequences * (sequence_length - 1)

    aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
    successes, points_of_failure, persistence_times = aux
    
    # Gather the resources
    aux_success = comm.gather(successes, root=0)

    if rank == 0:
        print('sl=', sl, 'ns=', ns)
        success = sum(aux_success, [])
        p_small = np.sum(success) / (number_of_sequences * trials)
        p_small_dic[(sl, ns)] = p_small
    
     # Sequence of twos 
    sequence_length = 2
    number_of_sequences = n_transitions
    aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
    successes, points_of_failure, persistence_times = aux
    
    # Gather the resources again
    aux_success = comm.gather(successes, root=0)
    
    if rank == 0:
        success = sum(aux_success, [])
        p2 = np.sum(success) / (number_of_sequences * trials)
        p_diff = p_small - p2**(sequence_length_large - 1)
        p2_dic[(sl, ns)] = p2
        p_diff_dic[(sl, ns)] = p_diff
        print(f'p_diff = {p_diff:.2f},  p2 = {p2:.2f} psmall = {p_small:.2f}, p_power = {p2**(sequence_length_large - 1):.2f}')
        print('=============')
 
        
        
# Store data as a pickle
if rank == 0:
    save_dic = {'p_diff': p_diff_dic, 'p2':p2_dic, 'p_small': p_small_dic, ''
               'hypercolumns': hypercolumns, 'minicolumns': minicolumns, 'number_of_sequences':number_of_sequences_vector, 
                'sequence_length': sequence_lengths_vector}
    
    filename = './data_length_sequences.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)