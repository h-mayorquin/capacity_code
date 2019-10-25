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
total_trials = 500
trials_per_rank = ceil(total_trials / size)

pattern_seed = rank
aux = serial_wrapper(trials_per_rank, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
successes, points_of_failure, persistence_times = aux

print(np.mean(successes))
collection = comm.gather(successes, root=0)
if rank == 0:
    print('rank', rank, np.mean(sum(collection, [])))


