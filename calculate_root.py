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
hypercolumns = 3
sequence_length = 2
number_of_sequences = 20
pattern_seed = rand()
desired_root = 0.9

# Read 
data_frame = pd.read_csv('../storage_capacity_data.csv', index_col=0)

# Do the calculations

aux = find_root_empirical(desired_root, trials, hypercolumns, minicolumns, sequence_length, pattern_seed, tolerance=0.01, verbose=False)

capacity, p_root, trials = aux
# Write
data_frame = data_frame.append({'hypercolumns':hypercolumns, 'minicolumns':minicolumns, 'sequence_length':sequence_length, 
                                'capacity':capacity, 'p_critical':p_root, 'trials':trials }, ignore_index=True)

# Store the data base
data_frame.to_csv('./storage_capacity_data.csv')
