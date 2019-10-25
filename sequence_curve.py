from mpi4py import MPI
import warnings
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
sequence_length = 2
number_of_sequences = 3
total_trials = 100
trials_per_rank = ceil(total_trials / size)
max_n_seq = floor((minicolumns - 1) ** 2 / sequence_length)
number_of_sequences_vector = np.arange(2, max_n_seq , 1)

if rank == 0:
    storage_dic_success = {}
    storage_dic_points_of_failure = {}
    storage_dic_persistent_times = {}
    
pattern_seed = rank
for ns in number_of_sequences_vector:

    aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
    successes, points_of_failure, persistence_times = aux
    
    aux_success = comm.gather(successes, root=0)
    aux_failure = comm.gather(points_of_failure, root=0)
    aux_times = comm.gather(persistence_times, root=0)

    if rank == 0:
        collection_of_persistent_times = 
        storage_dic_success[ns] = sum(aux_success, [])
        storage_dic_points_of_failure[ns] = sum(aux_failure, [])
        storage_dic_persistent_times = sum(aux_times, [])
        
        
# Store data as a pickle
if rank == 0:
    save_dic = {'success': storage_dic_success, 'points_of_failure':storage_dic_points_of_failure, 'persistent_times':storage_dic_persistent_times, 
               'hypercolumns': hypercolumns, 'minicolumns': minicolumns, 'number_of_sequences':number_of_sequences_vector, 
                'sequence_length': sequence_length}
    
    filename = './data.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(save_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)