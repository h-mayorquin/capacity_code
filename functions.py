from collections import deque
import numpy as np
import scipy as sp
import random
from math import ceil, floor
from copy import deepcopy
import multiprocessing as mp

from patterns_representation import PatternsRepresentation, build_network_representation


def get_beta(p):

    beta = np.log(p)

    return beta


def softmax(input_vector, G=1.0, minicolumns=2):
    """Calculate the softmax of a list of numbers w.

    Parameters
    ----------
    input_vector : the vector to softmax
    G : the constant for softmax, the bigger the G the more of a max it is

    Return
    ------
    a list of the same length as input_vectorof non-negative numbers

    Examples
    --------
    """

    # The lower bounds keeps the overflow from happening
    lower_bound = -600
    upper_bound = 600

    x = np.copy(input_vector)
    x_size = x.size
    x = np.reshape(x, (x_size // minicolumns, minicolumns))
    x = G * np.array(x)

    x[x < lower_bound] = lower_bound
    x[x > upper_bound] = upper_bound

    e = np.exp(x)
    dist = normalize_array(e)

    dist = np.reshape(dist, x_size)

    return dist


def normalize_array(array):
    """
    "Normalize an array over the second axis"

    :param array: the array to normalize
    :return: the normalized array
    """

    return array / np.sum(array, axis=1)[:, np.newaxis]


def strict_max(x, minicolumns):
    """
    A strict max that returns an array with 1 where the maximum of every minicolumn is
    :param x: the array
    :param minicolumns: number of minicolumns
    :return: the stric_max of the array
    """

    x = np.reshape(x, (x.size // minicolumns, minicolumns))
    z = np.zeros_like(x)
    maxes = np.argmax(x, axis=1)
    for max_index, max_aux in enumerate(maxes):
        z[max_index, max_aux] = 1

    return z.reshape(x.size)

##############
# Build P
################

def calculate_P_next(T1, T2, tau_z_pre, tau_z_post, Ts):
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    M1_pre = 1 - np.exp(-T1 / tau_z_pre)
    M2_pre = 1 - np.exp(-T2 / tau_z_pre)
    M1_post = 1 - np.exp(-T1 / tau_z_post)
    M2_post = 1 - np.exp(-T2 / tau_z_post)
    M1_p = 1 - np.exp(-T1 / tau_p)
    M2_p = 1 - np.exp(-T2 / tau_p)

    P_next = tau_z_pre * M1_pre * M2_pre - tau_p * M1_pre * M2_p + tau_p * M1_pre * M2_post * np.exp(-T2/tau_z_pre)
    P_next *= np.exp(-Ts / tau_z_pre)
    return P_next

def calculate_P_self_repeat(T1, tau_z_pre, tau_z_post, last_seen, Ts=0, memory=True):
    
    # Constants
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    M1_pre = 1 - np.exp(-T1 / tau_z_pre)
    M1_post = 1 - np.exp(-T1 / tau_z_post)
    M1_p = 1 - np.exp(-T1 / tau_p)
    
    if memory:
        m = M1_pre * np.exp(-(T1 + Ts) * last_seen / tau_z_pre)
        n = M1_post * np.exp(-(T1 + Ts) * last_seen / tau_z_post)
        r = (1 - np.exp(-T1 * last_seen / tau_p))
        
        P_self = T1 - tau_z_pre * (1 - m) * M1_pre - tau_z_post * (1 - n) * M1_post 
        P_self += tau_p * (1 - m) * (1 - n) * M1_p 
        P_self += tau_p * M1_pre * M1_post  * r
    else:
        m = M1_pre * 0
        n = M1_post * 0
        r = 1 - 0

        P_self = T1 - tau_z_pre * (1 - m) * M1_pre - tau_z_post * (1 - n) * M1_post 
        P_self += tau_p * (1 - m) * (1 - n) * M1_p         
        P_self += tau_p * M1_pre * M1_post * r
    return P_self 

def build_P(patterns, hypercolumns, minicolumns, tau_z_pre, tau_z_post, Tp, Ts, lower_bound=1e-6, verbose=False, memory=True):
    if verbose:
        print('Number of patterns you see before', number)

    if memory:
        buffer_size = int(np.ceil(-np.log(lower_bound) * (tau_z_pre / Tp))) 
    else:
        buffer_size = 1
    
    P = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    buffer = deque([], buffer_size)  # Holds up to three numbers
    last_seen_vector = np.zeros(minicolumns * hypercolumns)
    running_index = 0

    patterns_copy = list(patterns)[::-1]
    patterns_copy = list(patterns)
    while(len(patterns_copy) > 0):
        pattern = patterns_copy.pop()
 
        if verbose:
            print(patterns_copy)
            print(buffer)
            print('pattern', pattern)
            print('-----------')
            
        # Update the self patterns
        pattern_in_coordinates = [x  +  hypercolumn * minicolumns for (hypercolumn, x) in enumerate(pattern)]
        coordinate_pairs = [(x, y) for x in pattern_in_coordinates for y in pattern_in_coordinates] 
        
        for from_pattern, to_pattern in coordinate_pairs:
            last_seen = last_seen_vector[from_pattern]

            if last_seen == running_index:
                last_seen  = 1e10
                
            P[from_pattern, to_pattern] += calculate_P_self_repeat(Tp, tau_z_pre, tau_z_post,
                                                                   last_seen, memory=memory)
        
        # Store the patterns that you just saw
        for element in pattern_in_coordinates:
            last_seen_vector[element] = 0 

        # Update the next patterns
        for index, past_pattern in enumerate(buffer):
            P_next = calculate_P_next(Tp, Tp, tau_z_pre, tau_z_post, Ts)
            P_next_reverse = calculate_P_next(Tp, Tp, tau_z_post, tau_z_pre, Ts)

            for hypercolumn_present, present_element in enumerate(pattern):
                for hypercolumn_past, past_element in enumerate(past_pattern):
                    from_pattern = past_element + hypercolumn_past * minicolumns
                    to_pattern = present_element + hypercolumn_present * minicolumns
                    P[from_pattern,  to_pattern] += P_next * np.exp(-index * Tp/tau_z_pre)
                    P[to_pattern, from_pattern] += P_next_reverse * np.exp(-index * Tp/tau_z_post)

        buffer.appendleft(pattern)
        running_index += 1
        last_seen_vector += 1.0

    return P

def generate_sequence_one_hypercolum(m, N):

    possible_elements = [x for x in range(m)]
    sequence = []
    first_element = random.choice(possible_elements)
    current_element = first_element
    sequence.append(current_element)
    while(len(sequence) < N):
        next_element = random.choice(possible_elements)
        if next_element != current_element:
            sequence.append(next_element)
            current_element = next_element

    return sequence

# This generates sequences without repetition wihin the same hypercolumn
def generate_sequence(h, m, N):
    
    sequences = []
    for _ in range(h):
        sequences.append(generate_sequence_one_hypercolum(m, N))
    return np.array(sequences).T

def calculate_probabililties(patterns, minicolumns):
    hypercolumns = patterns.shape[1]
    n_patterns = patterns.shape[0]
    probabilities = np.zeros(minicolumns * hypercolumns)
    for minicolumn in range(minicolumns):
        probability_pattern = (patterns == minicolumn).sum(axis=0) 
        for hypercolumn, value in enumerate(probability_pattern):
            coordinate = minicolumn + hypercolumn * minicolumns
            probabilities[coordinate] = value
            
    return probabilities



def activity_to_neural_pattern(pattern, minicolumns):
    network_representation = np.zeros(len(pattern) * minicolumns)
    for hypercolumn_index, minicolumn_index in enumerate(pattern):
        index = hypercolumn_index * minicolumns + minicolumn_index
        network_representation[index] = 1
        
    return network_representation

def neural_pattern_to_activity(neural_pattern, minicolumns):
    return [x % minicolumns for x in np.where(neural_pattern == 1)[0]]


def build_dictionary_of_patterns(patterns_to_train, minicolumns):

    network_representation = build_network_representation(patterns_to_train, minicolumns)
    # This would be required in case there are repeated sequences
    aux, indexes = np.unique(network_representation, axis=0, return_index=True)
    indexes.sort()
    patterns_dic = {integer_index: network_representation[index] for (integer_index, index) in enumerate(indexes)}
    
    return patterns_dic



def calculate_patterns_timings(winning_patterns, dt, remove=0):
    """

    :param winning_patterns: A vector with the winning pattern for each point in time
    :param dt: the amount that the time moves at each step
    :param remove: only add the patterns if they are bigger than this number, used a small number to remove fluctuations

    :return: pattern_timins, a vector with information about the winning pattern, how long the network stayed at that
     configuration, when it got there, etc
    """

    # First we calculate where the change of pattern occurs
    change = np.diff(winning_patterns)
    indexes = np.where(change != 0)[0]

    # Add the end of the sequence
    indexes = np.append(indexes, len(winning_patterns) - 1)

    patterns = winning_patterns[indexes]
    patterns_timings = []

    previous = 0
    for pattern, index in zip(patterns, indexes):
        time = (index - previous + 1) * dt  # The one is because of the shift with np.change
        if time >= remove:
            patterns_timings.append((pattern, time, previous*dt, index * dt))
        previous = index

    return patterns_timings


##########################################################
###########################################################3
# Simulations functions
##########################################################
##########################################################


def serial_wrapper(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed,
                   tau_z_pre=0.050, sigma=0.0, tau_z_slow=0.005, tau_a=0.150, g_a=2.0, memory=True, 
                   recall_dynamics='normala', T_start=0.75, T_per_pattern=0.055, 
                   patterns_to_train=None):
    
    # Probably should be changed 
    tau_z_pre = tau_z_pre
    dt = 0.001
    
    # Trial parameters (change not very often)
    tau_z_post = 0.005
    training_time = 0.100
    epsilon = 1e-20
    remove = 0.010
    tau_s = 0.010
    g_a = g_a
    tau_a = tau_a
    sigma_in = sigma * np.sqrt(2 / tau_s)
    
    tau_z_fast = tau_z_pre
    #recall_dynamics = 'normal'  #('normala', 'one_tracea')
    
    T_training_total = training_time * number_of_sequences * sequence_length
    epsilon = dt/(T_training_total + dt)

    T_cue = tau_s
    T_recall = T_per_pattern * (sequence_length - 1) + T_start

    random.seed(pattern_seed)

    number_of_successes = []
    points_of_failure = []
    persistence_times = []
    pair_of_seq_and_recalled = []

    for _ in range(trials):
        aux = run_recall_trial(hypercolumns, minicolumns, number_of_sequences, 
                               sequence_length, dt, tau_z_pre, T_cue, T_recall, 
                               tau_z_post, training_time, remove, tau_s, g_a, tau_a, epsilon, 
                               memory, recall_dynamics, tau_z_slow, tau_z_fast, sigma_in,
                               patterns_to_train=patterns_to_train)
        
        correctly_recalled, points_of_failure_trial, persistence_times_trial, pairs = aux 

        # Append to lists
        n_recalled = sum(correctly_recalled)

        number_of_successes.append(n_recalled)
        points_of_failure.append(points_of_failure_trial)
        persistence_times.append(persistence_times_trial)
        pair_of_seq_and_recalled.append(pairs)
        
    return number_of_successes, points_of_failure, persistence_times, pair_of_seq_and_recalled


def run_recall_trial(hypercolumns, minicolumns, number_of_sequences, sequence_length, dt, 
                     tau_z_pre, T_cue, T_recall, tau_z_post, training_time, 
                     remove, tau_s, g_a, tau_a, epsilon, memory, recall_dynamics, tau_z_slow, tau_z_fast, 
                     sigma_in, patterns_to_train=None):
    
    # Random sequence of patterns
    n_patterns = number_of_sequences * sequence_length
    if patterns_to_train is None:
        patterns_to_train = generate_sequence(hypercolumns, minicolumns, n_patterns)
    
    # Calculate the weights and biases
    w, beta = create_w_and_beta(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, 
                                sequence_length, training_time, tau_z_pre, tau_z_post, epsilon, memory=memory)
    
    
    w_slow, beta_slow = create_w_and_beta(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, 
                                sequence_length, training_time, tau_z_slow, tau_z_post, epsilon, memory=memory)
    # Build a dictionary with all the patterns
    patterns_dic = build_dictionary_of_patterns(patterns_to_train, minicolumns)
    
    # Calculate the statitsics for the sequences
    aux = calculate_sequences_statistics(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, 
                                         T_cue, T_recall, dt, w, w_slow, beta, beta_slow, 
                                         tau_s, tau_a, g_a, patterns_dic, remove, recall_dynamics, 
                                         tau_z_slow, tau_z_fast, sigma_in)
    
    correctly_recalled, point_of_failure, persistence_times, seq_and_recalled_pairs = aux
    
    return correctly_recalled, point_of_failure, persistence_times, seq_and_recalled_pairs

def create_w_and_beta(patterns_to_train, hypercolumns, minicolumns,
                      number_of_sequences, sequence_length, training_time, 
                      tau_z_pre, tau_z_post, epsilon, memory=True, resting_time=0):

    Tp = training_time 
    Ts = 0
    P = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    for sequence_index in range(number_of_sequences):
        sequence = patterns_to_train.reshape((number_of_sequences, sequence_length, hypercolumns))[sequence_index, :]
        P += build_P(sequence, hypercolumns, minicolumns, tau_z_pre, tau_z_post, 
                     Tp, Ts, lower_bound=1e-6, verbose=False, memory=memory)
    
    T_training_total = Tp * number_of_sequences * sequence_length + resting_time
    value = Tp / T_training_total 
    p = calculate_probabililties(patterns_to_train, minicolumns) * value
    P /= T_training_total

    P[P < epsilon**2] = epsilon ** 2
    p[p < epsilon] = epsilon
    
    
    w = get_w_pre_post(P, p, p)
    beta = get_beta(p)
    
    return w, beta


def create_p_and_w(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, training_time, 
                      tau_z_pre, tau_z_post, epsilon, memory=True, resting_time=0):

    Tp = training_time 
    Ts = 0
    P = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    for sequence_index in range(number_of_sequences):
        sequence = patterns_to_train.reshape((number_of_sequences, sequence_length, hypercolumns))[sequence_index, :]
        P += build_P(sequence, hypercolumns, minicolumns, tau_z_pre, tau_z_post, 
                     Tp, Ts, lower_bound=1e-6, verbose=False, memory=memory)
    
    T_training_total = Tp * number_of_sequences * sequence_length + resting_time
    value = Tp  / T_training_total
    
    p = calculate_probabililties(patterns_to_train, minicolumns) * value
    P *= 1.0 / T_training_total
    P[P < epsilon**2] = epsilon ** 2
    p[p < epsilon] = epsilon
    
    w = get_w_pre_post(P, p, p, diagonal_zero=False)
    beta = get_beta(p)
    
    return w, beta, P, p


def get_w_pre_post(P, p_pre, p_post):

    outer = np.outer(p_post, p_pre)
    x = P / outer

    # P_qual zero and outer is bigger than epsilon
    #P_equal_zero = (P < epsilon) * (outer > epsilon)
    w = np.log(x)
    #w[P_equal_zero] = np.log10(epsilon)

    return w


def get_beta(p):

    beta = np.log(p)

    return beta

def calculate_sequences_statistics(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, 
                                   T_cue, T_recall, dt, w, w_slow, beta, beta_slow, tau_s, 
                                   tau_a, g_a, patterns_dic, remove, recall_dynamics, 
                                   tau_z_slow, tau_z_fast, sigma_in):
    
    correctly_recalled = []
    points_of_failure = []
    persistence_times = []
    reshaped_patterns = patterns_to_train.reshape((number_of_sequences, sequence_length, hypercolumns))
    sequences_to_store = []
    recalled_to_store = []
    
    for sequence_index in range(number_of_sequences):
        sequence = reshaped_patterns[sequence_index, :]
        sequences_to_store.append(sequence)

        aux = calculate_recalled_patterns(sequence, T_cue, T_recall, dt, w, w_slow, beta, 
                                          beta_slow, tau_s, tau_a, g_a, 
                                          patterns_dic, hypercolumns, minicolumns, remove, 
                                          recall_dynamics, tau_z_slow, tau_z_fast, sigma_in)

        recalled_patterns, T_per = aux
        persistence_times.append(T_per)
        recalled_to_store.append(recalled_patterns)
        
        # Get the persistent times
        if len(recalled_patterns) >= sequence_length:
            # This probably can be changed to check if the first point of failure is larger than sequence length
            correctly_recalled.append((sequence == recalled_patterns[:sequence_length]).all())
            first_point_of_failure = calculate_first_point_of_failure(sequence, recalled_patterns[:sequence_length], 'success')
        else:
            correctly_recalled.append(False)
            first_point_of_failure = calculate_first_point_of_failure(sequence[:len(recalled_patterns)], recalled_patterns[:sequence_length], 'too short')

        # For every sequences calculate the first point of failure
        points_of_failure.append(first_point_of_failure)
    
    return correctly_recalled, points_of_failure, persistence_times, (sequences_to_store, recalled_to_store)

def calculate_recalled_patterns(sequence, T_cue, T_recall, dt, w, w_slow, beta, 
                                beta_slow, tau_s, tau_a, g_a, patterns_dic, hypercolumns, minicolumns, remove, 
                                recall_dynamics, tau_z_slow, tau_z_fast, sigma_in):
    sequence_cue = sequence[0]
    
    winners = run_network_recall(sequence_cue, T_cue, T_recall, dt, w, 
                                 w_slow, beta, beta_slow, tau_s, tau_a, g_a, patterns_dic, hypercolumns, minicolumns, 
                                 recall_dynamics, tau_z_slow, tau_z_fast, sigma_in)
    timings = calculate_patterns_timings(winners, dt, remove=remove)

    # Get the list of the recalled patterns
    nr_list = [patterns_dic[x[0]] for x in timings]
    recalled_patterns = [[x % minicolumns for x in np.where(neural_representation == 1)[0]] for neural_representation in nr_list]

    # Get the persistent times (exluding the first as it comes from the cue)
    persistence_times = [x[1] for x in timings]
    
    return recalled_patterns, persistence_times

def calculate_first_point_of_failure(correct_sequence, recalled_sequence, failure_string):
    matching_vector = np.prod(correct_sequence == recalled_sequence, axis=1)
    points_of_failure = np.where(matching_vector == 0)[0]
    if points_of_failure.sum() > 0:
        first_point_of_failure = np.min(np.where(matching_vector == 0)[0])
    else:
        first_point_of_failure = failure_string

    return first_point_of_failure

def run_network_recall(sequence_cue, T_cue, T_recall, dt, w, w_slow, beta, beta_slow, tau_s,
                       tau_a, g_a, patterns_dic, hypercolumns, minicolumns, 
                       recall_dynamics, tau_z_slow, tau_z_fast, sigma_in):

    nt_cue = int(T_cue / dt)
    nt_recall = int(T_recall / dt)

    
    I_cue = activity_to_neural_pattern(sequence_cue, minicolumns)

    n_units = hypercolumns * minicolumns
    o = np.full(shape=n_units, fill_value=0.0)
    s = np.full(shape=n_units, fill_value=0.0)
    z_slow = np.full(shape=n_units, fill_value=0.0)
    z_fast = np.full(shape=n_units, fill_value=0.0)

    a = np.full(shape=n_units, fill_value=0.0)
    I = np.full(shape=n_units, fill_value=0.0)
    
    noise_vector =  sigma_in * np.sqrt(dt) * np.random.normal(0, 1.0, size=(nt_recall, n_units))

    winners = np.zeros(nt_recall + nt_cue)
    g_I = 10.0
    for i in range(nt_cue):
        # Step ahead
        noise = 0
        o, s, a, z_slow, z_fast = update_continuous(dt, tau_s, tau_a, g_a, w, w_slow, beta, beta_slow, 
                                                    g_I, I_cue, s, o, a, z_slow, z_fast, 
                                                    hypercolumns, minicolumns, recall_dynamics, tau_z_fast, tau_z_slow, noise)
        # Calculate winner
        winner = calculate_step_winner(o, patterns_dic)
        # Store winners
        winners[i] = winner

    g_I = 0.0
    for i in range(nt_recall):
        # Step ahead
        noise = noise_vector[i]
        o, s, a, z_slow, z_fast = update_continuous(dt, tau_s, tau_a, g_a, w, w_slow, beta, 
                                                    beta_slow, g_I, I_cue, s, o, a, z_slow, z_fast, 
                                                    hypercolumns, minicolumns, recall_dynamics, tau_z_fast, tau_z_slow, noise)
        # Calculate winner
        winner = calculate_step_winner(o, patterns_dic)
        # Store winners
        winners[i + nt_cue] = winner
        
    return winners


def update_continuous(dt, tau_s, tau_a, g_a, w, w_slow, beta, beta_slow, g_I, I, s, o, a, z_slow, z_fast,
                      hypercolumns, minicolumns, recall_dynamics, tau_z_fast, tau_z_slow, noise):
    
    # Calculate currents
    factor = 1
    if recall_dynamics[:-2] == 'normal':
        i = w @ o / hypercolumns
    if recall_dynamics[:-2] == 'one_trace':
        i = w @ z_fast / hypercolumns
    if recall_dynamics[:-2] == 'two_traces':
        i = (w @ z_fast + w_slow @ z_slow) / (2 * hypercolumns)
        factor = 2
    
    s += (dt / tau_s)  * (i  # Current
                       + beta  # Bias
                       + g_I * I  # Input current
                       - factor * g_a * a  # Adaptation
                       - s)  # s follow all of the s above
    # Add noise
    s += noise
    
    # Non-linearity
    if True:
        o = strict_max(s, minicolumns=minicolumns)
    else:
        o = softmax(s, G=G, minicolumns=minicolumns)

    # Update the adaptation
    if recall_dynamics[-1] == 'a':
        a += (dt / tau_a) * (o - a)
    else:
        a+=(dt / tau_a) * (o - a) * o + (dt / tau_s) * (o - a)*(1 - o)
   
    
    # Update z variables
    if recall_dynamics[:-2] == 'one_trace':
        z_fast += (dt / tau_z_fast) * (o - z_fast)
    if recall_dynamics[:-2] == 'two_traces':
        z_fast += (dt / tau_z_fast) * (o - z_fast)
        z_slow += (dt / tau_z_slow) * (o - z_slow)

    return o, s, a, z_slow, z_fast

def calculate_step_winner(o, patterns_dic):
    nominator = [np.dot(o, patterns_dic[pattern_index]) for pattern_index in patterns_dic.keys()]
    denominator = [np.linalg.norm(o) * np.linalg.norm(patterns_dic[pattern_index]) for pattern_index in patterns_dic.keys()]
    dis = [a / b for (a, b) in zip(nominator, denominator)]
    
    return np.argmax(dis)

#################################
# Root finding functions
#################################
def calculate_empirical_probability(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed, verbose=False):
    
    nprocs = mp.cpu_count()
    if trials > nprocs:
        trials_per_core = trials / nprocs 
        trials_list = [ceil(trials_per_core) for i in range(nprocs)]
        trials = sum(trials_list)
        seeds = [(i + 1) * (pattern_seed + 1) for i in range(nprocs)]
    else:
        nprocs = trials
        trials_per_core = 1
        trials_list = [1 for i in range(nprocs)]
        trials = sum(trials_list)
        seeds = [(i + 1) * (pattern_seed + 1) for i in range(nprocs)]
    print('nprocs', nprocs)
    print('trials', trials)
    
    pool = mp.Pool(processes=nprocs) 
    parameter_tuple = [(trials_per_core, hypercolumns, minicolumns, number_of_sequences, sequence_length, seed) for (trials_per_core, seed) in zip(trials_list, seeds)]
    result = pool.starmap(serial_wrapper, parameter_tuple)
    pool.close()

    succcess = sum([x[0] for x in result],[])
    
    return sum(succcess) / (trials * number_of_sequences)

def get_initial_bounds(desired_root, hypercolumns, minicolumns, sequence_length, pattern_seed, tolerance, verbose=False):
    
    bound = 2
    p_estimated = 1.0
    
    # Keep increasing the new bound until you passed the root
    while(p_estimated > desired_root - tolerance):
        bound = 2 * bound
        ns = bound 
        trials = min(100, find_trials_required(ns, sigma=tolerance, p=desired_root))
        p_estimated = calculate_empirical_probability(trials, hypercolumns, minicolumns, 
                                                      ns, sequence_length, pattern_seed, verbose)
        
        if verbose:
            print('bound', bound)
            print('p estimated', p_estimated)
        
        
    return bound * 0.5, bound, p_estimated



def find_root_empirical(desired_root, hypercolumns, minicolumns, sequence_length, pattern_seed, tolerance=0.01, verbose=False):
    
    aux = get_initial_bounds(desired_root, hypercolumns, minicolumns, sequence_length, pattern_seed, tolerance, verbose=verbose)
    left_bound, right_bound, p_estimated = aux
    
    calculate = True
    if abs(p_estimated - desired_root) < tolerance:
        p_root = p_estimated
        middle = right_bound
        calculate = False
    
    while(calculate):
        middle = floor((left_bound + right_bound) * 0.5)
        
        
        trials = max(3, min(100, find_trials_required(middle, sigma=tolerance, p=desired_root)))
        p = calculate_empirical_probability(trials, hypercolumns, minicolumns, 
                                            middle, sequence_length, pattern_seed, verbose)
        
        difference = p - desired_root

        if verbose:
            print('--------------')
            print('left bound', left_bound)
            print('right bound', right_bound)
            print('middle', middle)
            print('p', p)
            print('desired p', desired_root)
            print('difference', difference)
            print('trials', trials)

        if abs(difference) < tolerance:
            if verbose:
                print('found')
            p_root = p
            break

        if difference < 0.0:  # If p is to the smaller than the desired root   (-------p--*-----)
            right_bound = middle  # Move the right bound to the middle as the middle is too high

        if difference > 0.0:  # If p is larger than the desired root (---*--p-------)
            left_bound = middle   # Move the left bound to the middle as the middle is too low 

        if abs(left_bound - right_bound) < 1.5:
            if difference > 0:   # If p estimated is larger than desired go further to the right
                middle = right_bound
                p = calculate_empirical_probability(trials, hypercolumns, minicolumns, 
                                                    middle, sequence_length, pattern_seed)
            else:
                middle = left_bound
                p = calculate_empirical_probability(trials, hypercolumns, minicolumns, 
                                                    middle, sequence_length, pattern_seed)
            p_root = p
            break

    return middle, p_root, trials


def find_trials_required(number_of_sequences, sigma, p=0.9):
    """
    This gives the number of trials required for give a certain sigma.
    With this you can set sigma equal to the value that you want to get as standard deviation
    
    sigma = 0.01 means that the standard deviation would be 0.01
    
    It is better to get sigma = value / 3  to assure yourself that the 3 sigma value (99% per cent approximately)
    of the stimations would be within 0.01 of the estimated value of p = 0.9
    """
    return ceil((p * (1 - p)) / (number_of_sequences * sigma ** 2) - (1 / number_of_sequences))


def calculate_p_std(trials, n_s, p=0.9):
    mean_success = trials * n_s * p
    a = mean_success 
    b = trials * n_s - mean_success
    var_analytical = (a * b) /((a + b + 1) * (a + b) **2) 
    return np.sqrt(var_analytical)

def produce_pairs_with_constant_number_of_patterns(n_patterns):
    sl_old = n_patterns
    ns = 1
    pairs = []
    while(sl_old > 1):
        sl_new = floor(n_patterns / ns)
        if sl_new != sl_old:
            pairs.append((ns - 1, sl_old, sl_old * (ns - 1)))
        ns += 1
        sl_old = sl_new
        
    return pairs