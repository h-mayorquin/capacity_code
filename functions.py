from collections import deque
import numpy as np
import scipy as sp
import random
from math import ceil, floor
from copy import deepcopy

from network import Protocol, NetworkManager, Network
from patterns_representation import PatternsRepresentation
from analysis_functions import calculate_persistence_time, calculate_recall_quantities, calculate_triad_connectivity
from plotting_functions import plot_weight_matrix, plot_network_activity_angle, plot_persistent_matrix
from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances, calculate_patterns_timings
from connectivity_functions import get_w_pre_post, get_beta


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



def load_artificial_training(manager, patterns_to_train, hypercolumns, minicolumns, ns, sl, Tp):
    
    Ts = 0
    P = np.zeros((manager.nn.minicolumns * manager.nn.hypercolumns, manager.nn.minicolumns * manager.nn.hypercolumns))
    for sequence_index in range(ns):
        sequence = patterns_to_train.reshape((ns, sl, manager.nn.hypercolumns))[sequence_index, :]
        P += build_P(sequence, manager.nn.hypercolumns, manager.nn.minicolumns, manager.nn.tau_z_pre, manager.nn.tau_z_post, 
                     Tp, Ts, lower_bound=1e-6, verbose = False)

    value = (Tp ) / (Tp * (ns * sl))
    p = calculate_probabililties(patterns_to_train, minicolumns) * value

    manager.nn.P = P
    manager.nn.p_pre = p 
    manager.nn.p_post = p

    w = get_w_pre_post(manager.nn.P, manager.nn.p_pre, manager.nn.p_post, manager.nn.epsilon, diagonal_zero=False)
    beta = get_beta(manager.nn.p_pre, manager.nn.epsilon)

    manager.nn.w = w
    manager.nn.beta = beta

    values_to_save = ['o']
    manager.saving_dictionary = manager.get_saving_dictionary(values_to_save)


def calculate_first_point_of_failure(correct_sequence, recalled_sequence, failure_string):
    matching_vector = np.prod(correct_sequence == recalled_sequence, axis=1)
    points_of_failure = np.where(matching_vector == 0)[0]
    if points_of_failure.sum() > 0:
        first_point_of_failure = np.min(np.where(matching_vector == 0)[0])
    else:
        first_point_of_failure = failure_string

    return first_point_of_failure



def calculated_recalled_patterns(sequence, T_recall, T_cue, manager, remove=0.010):
    reset = True
    empty_history = True
    steady = True
    plasticity_on = False
    
    I_cue = activity_to_neural_pattern(sequence[0], manager.nn.minicolumns)
    manager.run_network_recall(T_recall=T_recall, I_cue=I_cue, T_cue=T_cue, reset=reset,
                               empty_history=empty_history, steady=steady, plasticity_on=plasticity_on)

    distances = calculate_angle_from_history(manager)
    winning = calculate_winning_pattern_from_distances(distances)
    timings = calculate_patterns_timings(manager, winning, remove=remove)

    nr_list = [manager.patterns_dic[x[0]] for x in timings]
    recalled_patterns = [[x % manager.nn.minicolumns for x in np.where(neural_representation == 1)[0]] for neural_representation in nr_list]
    
    return recalled_patterns, timings

def build_network_prototype(dt, hypercolumns, minicolumns, patterns, n_patterns, tau_z_pre, 
                            tau_z_post=0.005, training_times_base=0.100, ipi_base=0.0, random_instance=np.random):
    # General parameters
    epsilon = 10e-80
    strict_maximum = True
    tau_s = 0.010
    tau_a = 0.250
    g_I = 3.0
    g_a = 2.0
    G = 50.0
    sigma_out = 0.0
    
    n_patterns = patterns.shape[0]

    
    # Training protocol
    training_times = [training_times_base for i in range(n_patterns)]
    ipi_base = 0.000
    inter_pulse_intervals = [ipi_base for i in range(n_patterns)]
    inter_sequence_interval = 0.0
    resting_time = 0.0
    epochs = 1

    
    # Neural Network
    nn = Network(hypercolumns, minicolumns, G=G, tau_s=tau_s, tau_z_pre=tau_z_pre, tau_z_post=tau_z_post,
                     tau_a=tau_a, g_a=g_a, g_I=g_I, sigma_out=sigma_out, epsilon=epsilon, prng=random_instance,
                     strict_maximum=strict_maximum)

    values_to_save = []
    manager = NetworkManager(nn=nn, dt=dt, values_to_save=values_to_save)
    representation = PatternsRepresentation(patterns, minicolumns=minicolumns)

    # Build the protocol
    protocol = Protocol()
    protocol.simple_protocol(representation, training_times=training_times, inter_pulse_intervals=inter_pulse_intervals,
                        inter_sequence_interval=inter_sequence_interval, epochs=epochs, resting_time=resting_time)

    manager.update_patterns(protocol.network_representation)
    
    return manager


def run_memory_trial_serial(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, tau_z_pre, dt, tau_z_post, T_recall, T_cue, 
                            training_times, pattern_seed, recall_noise_instance=np.random, verbose=False):
    
    n_patterns = number_of_sequences * sequence_length
    successes = []
    persistence_times = []
    points_of_failure = []
    random.seed(pattern_seed)
    
    for trial_index in range(trials):
    
        if trial_index % 10 == 0 and verbose:
            print(trial_index)
        # Build random patterns
        
        patterns = generate_sequence(hypercolumns, minicolumns, n_patterns)

        # Build the network manage, returns the prototype
        manager = build_network_prototype(dt, hypercolumns, minicolumns, patterns, n_patterns, tau_z_pre, 
                                          training_times_base=training_times, random_instance=recall_noise_instance)

        # Load the artificial manager, this calculates the P and p and transforms them in w
        load_artificial_training(manager, patterns, hypercolumns, minicolumns, number_of_sequences, sequence_length, training_times)

        correctly_recalled = []
        point_of_failure = []

        reshaped_patterns = patterns.reshape((number_of_sequences, sequence_length, hypercolumns))
        for sequence_index in range(number_of_sequences):
            sequence = reshaped_patterns[sequence_index, :]

            recalled_patterns, timings = calculated_recalled_patterns(sequence, T_recall, T_cue, manager)
            persistence_times += [x[1] for x in timings][1:]

            if len(recalled_patterns) >= sequence_length:
                # This probably can be changed to check if the first point of failure is larger than sequence length
                correctly_recalled.append((sequence == recalled_patterns[:sequence_length]).all())
                first_point_of_failure = calculate_first_point_of_failure(sequence, recalled_patterns[:sequence_length], 'success')
            else:
                correctly_recalled.append(False)
                first_point_of_failure = calculate_first_point_of_failure(sequence[:len(recalled_patterns)], recalled_patterns[:sequence_length], 'too short')

            # For every sequences calculate the first point of failure
            point_of_failure.append(first_point_of_failure)

        number_of_successes = np.sum(correctly_recalled)
        successes.append(number_of_successes)
        points_of_failure += point_of_failure
        
    return successes, points_of_failure, persistence_times

def serial_wrapper(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, pattern_seed):
    verbose = False
    dt = 0.001
    tau_z_pre = 0.050
    tau_z_post = 0.005
    
    # Training protocol
    Tp = 0.100

    # Recall times
    T_recall = (sequence_length - 1) * 0.050
    T_cue = 0.050

    recall_noise_instance = np.random
    
    aux = run_memory_trial_serial(trials, hypercolumns, minicolumns, number_of_sequences, sequence_length, tau_z_pre, tau_z_post, dt,
                              T_recall, T_cue, Tp, pattern_seed, recall_noise_instance=recall_noise_instance, verbose=verbose)

    return aux