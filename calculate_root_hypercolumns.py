import warnings
import pickle 
import pandas as pd
import numpy as np
import random
from math import ceil, floor
from copy import deepcopy
from functions import *

warnings.filterwarnings('ignore')

minicolumns = 8
hypercolumns = 15
sequence_length = 2
number_of_sequences = 20
desired_root = 0.9
verbose = True


# Do the calculations
hypercolumns_set = [25, 30, 40, 45]
for hypercolumns in hypercolumns_set:
    print('hypercolumns', hypercolumns)
    print('minicolumns', minicolumns)
    pattern_seed = np.random.randint(0, 20)
    aux = find_root_empirical(desired_root, hypercolumns, minicolumns, sequence_length, pattern_seed, tolerance=0.01, verbose=verbose)

    capacity, p_root, trials = aux

    # Read 
    data_frame = pd.read_csv('../storage_capacity_data.csv', index_col=0)

    # Write
    data_frame = data_frame.append({'hypercolumns':hypercolumns, 'minicolumns':minicolumns, 'sequence_length':sequence_length, 
                                    'capacity':capacity, 'p_critical':p_root, 'trials':trials }, ignore_index=True)

    # Store the data base
    data_frame.to_csv('../storage_capacity_data.csv')
    print('Stored')
    print('================')