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

print('Hello from process {} out of {}'.format(rank, size))

# Run a dummy example
hypercolumns = 5
minicolumns = 10
sequence_length = 2
number_of_sequences = 3
trials = 5

succesess = [i for i in range(trials)]

#pattern_seed = rank
#aux = serial_wrapper(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
#successes, points_of_failure, persistence_times = aux

#print(np.mean(successes))
if rank == 0:
    collection = comm.gather(succesess, root=0)
    print(len(collection))


