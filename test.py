import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import random
from math import ceil, floor
from copy import deepcopy
from functions import *

# Run a dummy example
hypercolumns = 5
minicolumns = 10
sequence_length = 4
number_of_sequences = 5
trials = 100

nprocs = mp.cpu_count() - 1
trials_per_core = trials / nprocs 
trials_list = [ceil(trials_per_core) for i in range(nprocs)]
seeds = [i for i in range(nprocs)]

pool = mp.Pool(processes=nprocs)
parameter_tuple = [(trials_per_core, hypercolumns, minicolumns, number_of_sequences, sequence_length, seed) for (trials_per_core, seed) in zip(trials_list, seeds)]
result = pool.starmap(serial_wrapper, parameter_tuple)
pool.close()

succcess = sum([x[0] for x in result],[])
points_of_failure = sum([x[1] for x in result], [])
persistent_times = sum([x[2] for x in result], [])
filtered_failures = [x for x in points_of_failure if x not in ['success', 'too short']]

#pattern_seed = 1
#aux = serial_wrapper(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed)
#successes, points_of_failure, persistence_times = aux

print(np.mean(succcess))
