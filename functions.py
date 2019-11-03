from collections import deque
import numpy as np
import scipy as sp
import random
from math import ceil, floor
from copy import deepcopy

from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation, build_network_representation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities, calculate_triad_connectivity
from plotting_functions import plot_weight_matrix, plot_network_activity_angle, plot_persistent_matrix
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances, calculate_patterns_timings
from connectivity_functions import get_w_pre_post, get_beta, strict_max


def calculate_P_self(T1, tau_z_pre, tau_z_post):
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    M1_pre = 1 - np.exp(-T1 / tau_z_pre)
    M1_post = 1 - np.exp(-T1 / tau_z_post)
    M1_p = 1 - np.exp(-T1 / tau_p)
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    
    P_self = T1 - tau_z_pre * M1_pre - tau_z_post * M1_post + tau_p * M1_p + tau_p * M1_pre * M1_post 
    return P_self 

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

def calculate_P_self_repeat(T1, tau_z_pre, tau_z_post, last_seen, Ts=0):
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    M1_pre = 1 - np.exp(-T1 / tau_z_pre)
    M1_post = 1 - np.exp(-T1 / tau_z_post)
    M1_p = 1 - np.exp(-T1 / tau_p)
    tau_p = (tau_z_pre * tau_z_post) / (tau_z_pre + tau_z_post)
    
    m = M1_pre * np.exp(-(T1 + Ts) * last_seen / tau_z_pre)
    n = M1_post * np.exp(-(T1 + Ts) * last_seen / tau_z_post)
    
    P_self = T1 - tau_z_pre * (1 - m) * M1_pre - tau_z_post * (1 - n) * M1_post 
    P_self += tau_p * (1 - m) * (1 - n) * M1_p 
    P_self += tau_p * M1_pre * M1_post * (1 - 0.0 * np.exp(-T1 * last_seen / tau_p)) 
    
    return P_self 


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


def build_P(patterns, hypercolumns, minicolumns, tau_z_pre, tau_z_post, Tp, Ts, lower_bound=1e-6, verbose = False):
    if verbose:
        print('Number of patterns you see before', number)

    number = int(np.ceil(-np.log(lower_bound) * (tau_z_pre / Tp))) 
    P = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    buffer = deque([], number)  # Holds up to three numbers
    last_seen_vector = np.zeros(minicolumns * hypercolumns)
    running_index = 0

    patterns_copy = list(patterns)[::-1]
    patterns_copy = list(patterns)
    while(len(patterns_copy) > 0):
        pattern = patterns_copy.pop()
        last_seen_vector += 1.0

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
                
            P[from_pattern, to_pattern] += calculate_P_self_repeat(Tp, tau_z_pre, tau_z_post, last_seen)
        
        # Increase counter for seen patterns
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
                    P[to_pattern, from_pattern] += P_next_reverse * np.exp(-index * Tp/tau_z_pre)

        buffer.appendleft(pattern)
        running_index += 1
        
    return P

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




def calculate_first_point_of_failure(correct_sequence, recalled_sequence, failure_string):
    matching_vector = np.prod(correct_sequence == recalled_sequence, axis=1)
    points_of_failure = np.where(matching_vector == 0)[0]
    if points_of_failure.sum() > 0:
        first_point_of_failure = np.min(np.where(matching_vector == 0)[0])
    else:
        first_point_of_failure = failure_string

    return first_point_of_failure

def create_w_and_beta(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, training_time, tau_z_pre, tau_z_post, epsilon):

    Tp = training_time 
    Ts = 0
    P = np.zeros((minicolumns * hypercolumns, minicolumns * hypercolumns))
    for sequence_index in range(number_of_sequences):
        sequence = patterns_to_train.reshape((number_of_sequences, sequence_length, hypercolumns))[sequence_index, :]
        P += build_P(sequence, hypercolumns, minicolumns, tau_z_pre, tau_z_post, 
                     Tp, Ts, lower_bound=1e-6, verbose = False)

    value = (Tp ) / (Tp * (number_of_sequences * sequence_length))
    p = calculate_probabililties(patterns_to_train, minicolumns) * value

    w = get_w_pre_post(P, p, p, epsilon, diagonal_zero=False)
    beta = get_beta(p, epsilon)
    
    return w, beta

def build_dictionary_of_patterns(patterns_to_train, minicolumns):

    network_representation = build_network_representation(patterns_to_train, minicolumns)
    # This would be required in case there are repeated sequences
    aux, indexes = np.unique(network_representation, axis=0, return_index=True)
    indexes.sort()
    patterns_dic = {integer_index: network_representation[index] for (integer_index, index) in enumerate(indexes)}
    
    return patterns_dic


def update_continuous(dt, tau_s, tau_a, g_a, w, beta, g_I, I, s, o, a, hypercolumns, minicolumns):
    
    # Calculate currents
    i = w @ o / hypercolumns
    s += (dt / tau_s)  * (i  # Current
                       + beta  # Bias
                       + g_I * I  # Input current
                       - g_a * a  # Adaptation
                       - s)  # s follow all of the s above
    # Non-linearity
    if True:
        o = strict_max(s, minicolumns=minicolumns)
    else:
        o = softmax(s, G=G, minicolumns=minicolumns)

    # Update the adaptation
    a += (dt / tau_a) * (o - a)

    return o, s, a

def calculate_step_winner(o, patterns_dic):
    nominator = [np.dot(o, patterns_dic[pattern_index]) for pattern_index in patterns_dic.keys()]
    denominator = [np.linalg.norm(o) * np.linalg.norm(patterns_dic[pattern_index]) for pattern_index in patterns_dic.keys()]
    dis = [a / b for (a, b) in zip(nominator, denominator)]
    
    return np.argmax(dis)
    


def run_network_recall(sequence_cue, T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, patterns_dic, hypercolumns, minicolumns):

    nt_cue = int(T_cue / dt)
    nt_recall = int(T_recall / dt)

    
    I_cue = activity_to_neural_pattern(sequence_cue, minicolumns)

    n_units = hypercolumns * minicolumns
    o = np.full(shape=n_units, fill_value=0.0)
    s = np.full(shape=n_units, fill_value=0.0)

    a = np.full(shape=n_units, fill_value=0.0)
    I = np.full(shape=n_units, fill_value=0.0)
    
    winners = np.zeros(nt_cue + nt_recall)
    g_I = 10.0
    for i in range(nt_cue):
        # Step ahead
        o, s, a = update_continuous(dt, tau_s, tau_a, g_a, w, beta, g_I, I_cue, s, o, a, hypercolumns, minicolumns)
        # Calculate winner
        winner = calculate_step_winner(o, patterns_dic)
        # Store winners
        winners[i] = winner

    g_I = 0.0
    for i in range(nt_recall):
        # Step ahead
        o, s, a = update_continuous(dt, tau_s, tau_a, g_a, w, beta, g_I, I_cue, s, o, a, hypercolumns, minicolumns)
        # Calculate winner
        winner = calculate_step_winner(o, patterns_dic)
        # Store winners
        winners[i + nt_cue] = winner
        
    return winners


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


def calculate_recalled_patterns(sequence, T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, patterns_dic, hypercolumns, minicolumns, remove):
    sequence_cue = sequence[0]
    
    winners = run_network_recall(sequence_cue, T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, patterns_dic, hypercolumns, minicolumns)
    timings = calculate_patterns_timings(winners, dt, remove=remove)

    # Get the list of the recalled patterns
    nr_list = [patterns_dic[x[0]] for x in timings]
    recalled_patterns = [[x % minicolumns for x in np.where(neural_representation == 1)[0]] for neural_representation in nr_list]

    # Get the persistent times (exluding the first as it comes from the cue)
    persistence_times = [x[1] for x in timings][1:]
    
    return recalled_patterns, persistence_times

def calculate_sequences_statistics(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, 
                                   T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, patterns_dic, remove):
    
    correctly_recalled = []
    points_of_failure = []
    persistence_times = []
    reshaped_patterns = patterns_to_train.reshape((number_of_sequences, sequence_length, hypercolumns))
    
    for sequence_index in range(number_of_sequences):
        sequence = reshaped_patterns[sequence_index, :]

        aux = calculate_recalled_patterns(sequence, T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, 
                                          patterns_dic, hypercolumns, minicolumns, remove)

        recalled_patterns, T_per = aux
        persistence_times += T_per
        
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
    
    return correctly_recalled, points_of_failure, persistence_times

def run_recall_trial(hypercolumns, minicolumns, number_of_sequences, sequence_length, dt, tau_z_pre, T_cue, T_recall,
                     tau_z_post, training_time, remove, tau_s, g_a, tau_a, epsilon):
    
    # Random sequence of patterns
    n_patterns = number_of_sequences * sequence_length
    patterns_to_train = generate_sequence(hypercolumns, minicolumns, n_patterns)
    # Calculate the weights and biases
    w, beta = create_w_and_beta(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, 
                                sequence_length, training_time, tau_z_pre, tau_z_post, epsilon)
    # Build a dictionary with all the patterns
    patterns_dic = build_dictionary_of_patterns(patterns_to_train, minicolumns)
    
    # Calculate the statitsics for the sequences
    aux = calculate_sequences_statistics(patterns_to_train, hypercolumns, minicolumns, number_of_sequences, sequence_length, 
                                         T_cue, T_recall, dt, w, beta, tau_s, tau_a, g_a, patterns_dic, remove)
    
    correctly_recalled, point_of_failure, persistence_times = aux
    
    return correctly_recalled, point_of_failure, persistence_times


def serial_wrapper(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed):
    
    # Probably should be changed 
    tau_z_pre = 0.050
    dt = 0.001
    
    # Trial parameters (change not very often)
    tau_z_post = 0.005
    training_time = 0.100
    epsilon = 1e-20
    remove = 0.010
    tau_s = 0.010
    g_a = 2.0
    tau_a = 0.150

    T_cue = 0.050
    T_recall = 0.050 * (sequence_length - 1)

    random.seed(pattern_seed)

    number_of_successes = []
    points_of_failure = []
    persistence_times = []

    for _ in range(trials):
        aux = run_recall_trial(hypercolumns, minicolumns, number_of_sequences, sequence_length, dt, tau_z_pre, T_cue, T_recall,
                         tau_z_post, training_time, remove, tau_s, g_a, tau_a, epsilon)
        correctly_recalled, points_of_failure_trial, persistence_times_trial = aux 

        # Append to lists
        n_recalled = sum(correctly_recalled)

        number_of_successes.append(n_recalled)
        points_of_failure.append(points_of_failure_trial)
        persistence_times.append(persistence_times_trial)
        
    return number_of_successes, points_of_failure, persistence_times