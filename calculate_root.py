import warnings
import pickle 
import pandas as pd
import numpy as np
import random
from math import ceil, floor
from copy import deepcopy
from functions import *

warnings.filterwarnings('ignore')

minicolumns = 10
hypercolumns = 5
sequence_length = 2
number_of_sequences = 20
pattern_seed = np.random.randint(0, 20)
desired_root = 0.9
verbose = True

n_patterns = 100
pairs = produce_pairs_with_constant_number_of_patterns(n_patterns)[3:-3]
# Format is hypercolumns, minicolumns, extra
pairs = [(3, 66, 0)]
# Do the calculations
for pair in pairs:
    hypercolumns, minicolumns, extra = pair
    print('hypercolumns', hypercolumns)
    print('minicolumns', minicolumns)
    print('extra', extra)
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
