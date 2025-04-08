from concurrent.futures import thread
import src.utils.BooleanNetwork as BooleanNetwork
import os
import operator
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats
from threading import Thread
from queue import Queue, Empty
import time
import pandas as pd
import scripts.utils as utils
import scripts.io_scripts as io_scripts
import concurrent.futures

network_parameters_file = "/input/Network-parameters/"
save_chaos_test_path = "/input/Behaviour-results/"
posteva_folder = "/input/Post-evaluation/"
fpt_folder = "/input/Fpt-performance/"

### ------------------------------- Sensibility Functions ----------------------------------- ###

def normalizedHamingDistance(state_a, state_b, n):
    sum_dif = 0
    for i in range(n):
        sum_dif += abs((state_a[i] - state_b[i]))
    norm_hamming_dist = (1/float(n)) * (sum_dif)
    return norm_hamming_dist

def networkSensibilityAnalisys(network, max_steps, num_evaluations):
    print("Running Sensibility to initial condition test: time_steps=%d num_evaluation=%d (%dN)" % (max_steps, num_evaluations, network.num_nodes))
    average_delta = 0
    # network.printNetworkParameters()
    for i in range(num_evaluations):
        state_a = network.generateInitialState()
        all_state, _ = network.getNetworkStatesOverTimeSteps(max_steps)
        final_node_states_a = all_state[-1]
        # print(f"\nEstado inicial: {state_a}\nEstado Final: {final_node_states_a} ------")

        state_b = network.flipNodeinInitialState(1)
        all_state, _ = network.getNetworkStatesOverTimeSteps(max_steps)
        final_node_states_b = all_state[-1]
        # print(f"Estado inicial: {state_b}\nEstado Final: {final_node_states_b} --------")

        delta = normalizedHamingDistance(final_node_states_a, final_node_states_b, network.num_nodes) - normalizedHamingDistance(state_a, state_b, network.num_nodes)
        average_delta += delta
        # print(f"Delta = {delta}")
        # time.sleep(3)
        
    average_delta = (average_delta/float(num_evaluations))
    return average_delta

### ------------------------------- Step Lenght Functions ----------------------------------- ###

def calculateAverageStepLenght(network, num_evaluations, sim_time):
    # print("Measuring %d Average SL in %d seconds (%dN)" % (num_evaluations, sim_time, network.num_nodes))
    average_sl = 0
    for i in range(num_evaluations):
        average_sl += network.getAvgSLUntilSimulationEnd(sim_time)

    average_sl = (average_sl/num_evaluations)
    # print("%dN network has an average Step Length: %d cm(max of %d)" % (network.num_nodes, average_sl, pow(2, network.num_nodes/2)/32))
    
    return average_sl

### ------------------------------- Complexity Functions ----------------------------------- ###

def calculateFrequency(bn_states, num_nodes):
    states_string = ["".join(str(y) for y in x) for x in bn_states]
    count_states_list = list({x:states_string.count(x) for x in states_string}.values())
    frequency_list = [x/len(states_string) for x in count_states_list]
    # num_states_possible = 2**num_nodes
    # frequency_list += [0]* (num_states_possible - len(frequency_list))
    # print(len(frequency_list))

    return frequency_list

def calculateDisequilibrium(frequency_list):
    N = len(frequency_list)
    D = np.sum([((f - 1/N)**2) for f in frequency_list])
    return D

def LMC(bn_states, num_nodes):
    # print(f"Estados: {bn_states}")
    frequency_list = calculateFrequency(bn_states, num_nodes)
    # print(f"Frequencia: {frequency_list}")

    entropy = stats.entropy(frequency_list)
    disequilibrium = calculateDisequilibrium(frequency_list)
    complexity = entropy * disequilibrium

    # print(f"H(x) = {H}\nD(x) = {D}\nC(x) = H(x)*D(x) = {complexity}")
    return complexity, entropy, disequilibrium

def calculateNetworkComplexity(network, max_steps, num_evaluations):
    print("Running Complexity test for test: time_steps=%d num_evaluation=%d (%dN)" % (max_steps, num_evaluations, network.num_nodes))
    average_comp = 0
    average_entropy = 0
    average_diseq = 0
    for i in range(num_evaluations):
        state = network.generateInitialState()
        all_states, _ = network.getNetworkStatesOverTimeSteps(max_steps)
        # print(f"All states in Compelxity test:\n{all_states}")
        comp, entropy, diseq = LMC(all_states, network.num_nodes)
        average_comp += comp
        average_entropy += entropy
        average_diseq += diseq
        
    average_comp = (average_comp/num_evaluations)
    average_entropy = (average_entropy/num_evaluations)
    average_diseq = (average_diseq/num_evaluations)
    return average_comp, average_entropy, average_diseq

def runExperiment(network, max_steps, sim_time, num_evaluations):
    avg_delta = networkSensibilityAnalisys(network, max_steps, num_evaluations)
    avg_comp, avg_entropy, avg_diseq = calculateNetworkComplexity(network, max_steps, num_evaluations)
    avg_sl = calculateAverageStepLenght(network, num_evaluations, sim_time)
    # avg_sl = 10
    # avg_comp = avg_entropy = avg_diseq = 0

    network.setDeltaValue(avg_delta)
    network.setStepLength(avg_sl)
    network.setLMCValues(avg_comp, avg_entropy, avg_diseq)

    return network

### ------------------------------- Main Functions ----------------------------------- ###

def analyzeNetworkBehaviour(all_networks, num_nodes, bn_type, max_steps, num_evaluations, sim_time, num_threads):
    start_time = time.time()

    # # random.seed(10)
    # rbn = BooleanNetwork.BooleanNetwork(num_nodes, K=20)
    # networkSensibilityAnalisys(rbn, max_steps, num_evaluations)

    # ebns = utils.setEBNsFromFile(ebn_parameters_folder, num_nodes, 20)
    # all_delta = []
    # print("Starting %d initial states sensibility test for %d steps (%s N)" % (num_evaluations, max_steps, ','.join(map(str,num_nodes))))
    # for ebn in ebns:
    #     all_delta.append(networkSensibilityAnalisys(ebn, max_steps, num_evaluations))
    # plotPerformanceOverChaos(ebns, all_delta)

    print("Behaviour Analysis for %s %s N\n" % (bn_type, ','.join(map(str,num_nodes))))
    for node in num_nodes:
        networks = []
        label = "%s %d" % (bn_type, node)
        all_networks[label] = dict()
        if bn_type == "RBN":
            num_networks = 100
            networks = io_scripts.readRBNsParametersAndPerformance(network_parameters_file, node)
            networks = networks[:num_networks]
        else:
            num_robots = 20
            evolution_targets_type = "random-targets"
            networks = utils.createEBNsFromPostEva(posteva_folder, fpt_folder, node, num_robots, evolution_targets_type)

        print("\nStarting analysis for %s %s N" % (bn_type, str(node)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as tpe:
            futures = [tpe.submit(runExperiment, network, max_steps, sim_time, num_evaluations) for network in networks]
            for future in concurrent.futures.as_completed(futures):
                network = future.result()
                print("%dN network has an average Delta of %.2f (%s)\nAverage Complexity of %.2f, Entropy of %.2f and Disequilibrium of %.2f\nAverage Step Length: %d cm(max of %d)\n" % (network.num_nodes, network.average_delta, 
                    ("Ordered" if network.average_delta <= 0 else "Chaotic"), network.average_complexity, network.average_entropy, network.average_disequilibrium, network.total_average_sim_sl, pow(2, network.num_nodes/2)/32))

        networks = sorted(networks, key=operator.attrgetter('average_delta'))
        all_networks[label]["fpt"] = [network.fpt_result for network in networks]
        all_networks[label]["delta"] = [network.average_delta for network in networks]
        all_networks[label]["complexity"] = [network.average_complexity for network in networks]
        all_networks[label]["entropy"] = [network.average_entropy for network in networks]
        all_networks[label]["disequilibrium"] = [network.average_disequilibrium for network in networks]
        all_networks[label]["sl"] = [network.total_average_sim_sl for network in networks]
        io_scripts.savePerformanceOverChaos(save_chaos_test_path, all_networks[label], node, bn_type, max_steps, num_evaluations)

    print("Time running: %s" % (time.time() - start_time))
    return all_networks

### ----------------------------------------------------------------------------------------------------------- ###

def main(bn_type, num_threads, list_nodes):
    sim_time = 3000
    max_steps = 10000
    num_nodes = [int(x) for x in list_nodes.split(",") if x != '']
    num_evaluations = 75

    all_networks = dict()
    analyzeNetworkBehaviour(all_networks, num_nodes, bn_type, max_steps, num_evaluations, sim_time, num_threads)