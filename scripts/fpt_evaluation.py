import sys
import time
import random
import src.KilobotsSearchExperiment as KilobotsSearchExperiment
import src.utils.BooleanNetwork as BooleanNetwork
import src.utils.Targets as Targets
import scripts.log_script as LOG
import os
from operator import itemgetter
import scripts.io_scripts as io_scripts

# Simulation parameters
SAVE_POS = False
SAVE_PAR = False
KILOBOT_BIAS = True

# Input files
posteva_folder = "data/Post-evaluation/"

# Output files
fpt_results_folder = "data/Fpt-performance/"

#### ------------------------------------------------------------------------------------------------------ ####
def readMultiEBNs(num_nodes, num_robots, max_trials, arena_radius, experiment_date, evolution_targets_type):
    ebn_parameters = []
    for subdir, dirs, files in os.walk(posteva_folder):
        dirs.sort()
        for folder in dirs:
            elements = folder.split("_")
            file_date = ""
            file_nodes = -1
            file_robots = -1
            file_seed = -1
            file_targets = ""
            for e in elements:
                if e.endswith("N"):
                    file_nodes = int(e.replace("N", ""))
                if e.endswith("R"):
                    file_robots = int(e.replace("R", ""))
                if e.startswith("202"):
                    file_date = e  
                if e.endswith("seed"):
                    file_seed = int(e.replace("seed", ""))
                if e.endswith("targets"):
                    file_targets = e             

            if num_nodes != file_nodes or num_robots != file_robots:
                continue

            if evolution_targets_type != file_targets:
                continue
        
            if experiment_date != "":
                if not file_date.startswith(experiment_date):
                    continue

                # if file_date != experiment_date:
                #     continue

            # if file_seed not in [1,2,4,8,9,10]:
            #     continue

            print(folder)
            ebn_parameters.append(dict())
            ebn_parameters[-1]["date"] = file_date
            ebn_parameters[-1]["seed"] = file_seed
            ebn_parameters[-1]["target type"] = file_targets
            for file_name in os.listdir(posteva_folder + folder):
                if file_name == "initial-states.txt":
                    ebn_parameters[-1]["initial state"] = posteva_folder + folder + "/" + file_name
                if file_name == "parameters_best_ebn.txt":
                    ebn_parameters[-1]["parameters"] = posteva_folder + folder + "/" + file_name

    ebn_parameters = sorted(ebn_parameters, key=itemgetter('seed'))

    print("Loading EBNs:")
    ebns = []
    for ebn_info in ebn_parameters:
        print(ebn_info["parameters"])
        ebn = BooleanNetwork.BooleanNetwork(num_nodes, bn_type="ebn", read_from=ebn_info["parameters"], net_id=f'{ebn_info["seed"]:04}')
        ebn.readRobotsInitialStates(num_robots, ebn_info["initial state"])
        ebn.setPerformanceExperiment(num_robots, max_trials, round((arena_radius-0.0250)*200), KILOBOT_BIAS, save_exp=True, target_type=f"_id:{ebn_info['target type']}")
        ebns.append(ebn)

    return ebns

#### ------------------------------------------------------------------------------------------------------ ####
def performanceEvaluationForRBN(num_robots, all_nodes, arena_radius, simulation_time, n_threads):
    num_networks = 100
    evaluations = 20
    max_trials = 100
    num_threads = n_threads

    for num_nodes in all_nodes:
        random.seed(15)
        targets_position = Targets.createTargetPosition(max_trials, False, arena_radius)

        random.seed(15)
        rbns = []
        for count_net in range(num_networks):
            rbn = BooleanNetwork.BooleanNetwork(num_nodes, bn_type="RBN", net_id=f'{count_net:04}')
            rbn.createRobotsInitialStates(num_robots)
            rbn.setPerformanceExperiment(num_robots, max_trials, round((arena_radius-0.025)*200), KILOBOT_BIAS, save_exp=True)
            rbns.append(rbn)

        print("Starting performance evaluation for %dN RBN.\nRobots: %d - Arena Radius: %.3fcm\nNetworks: %d - Evaluations: %d - Trials: %d - Bias: %s"
            % (num_nodes, num_robots, arena_radius, num_networks, evaluations, max_trials, KILOBOT_BIAS))
        for count_eva in range(evaluations):
            print("\nStarting %d trials for the %d networks (%d evaluation of %d):" % (max_trials, num_networks, (count_eva+1), evaluations))
            experiment = KilobotsSearchExperiment.NetworkKilobotsExperiment(num_threads, num_robots, targets_position, arena_radius, simulation_time, KILOBOT_BIAS, SAVE_POS, SAVE_PAR)
            experiment.executeKilobotExperimentTrials(rbns)
            for rbn in rbns:
                rbn.experiment_performance.resetResults()

def performanceEvaluationForMultiEBNs(num_robots, all_nodes, arena_radius, simulation_time, num_threads):
    evaluations = 20
    max_trials = 100
    experiment_date = ""
    evolution_targets_type = "random-targets"

    for num_nodes in all_nodes:
        ebns = readMultiEBNs(num_nodes, num_robots, max_trials, arena_radius, experiment_date, evolution_targets_type)
        
        random.seed(15)
        targets_position = Targets.createTargetPosition(max_trials, False, arena_radius)

        print(f"Starting {evaluations} performance evaluations for {len(ebns)} seeds of {num_nodes}N EBNs ({num_robots} robots)({max_trials} trials)")
        for count_eva in range(evaluations):
            print(f"\nStarting {max_trials} trials for {len(ebns)} seeds of {num_nodes} EBNs ({count_eva+1} evaluation of {evaluations})")
            experiment = KilobotsSearchExperiment.NetworkKilobotsExperiment(num_threads, num_robots, targets_position, arena_radius, simulation_time, KILOBOT_BIAS, SAVE_POS, SAVE_PAR)
            experiment.executeKilobotExperimentTrials(ebns)
            for ebn in ebns:
                ebn.experiment_performance.resetResults()

def runFptEvaluation(num_threads, strategy, list_nodes):
    start_time = time.time()

    sim_options = io_scripts.readSimulationConfigFile()
    num_nodes = [int(x) for x in list_nodes.split(",") if x != '']

    if strategy == "RBN":
        performanceEvaluationForRBN(sim_options['num_robots'], num_nodes, sim_options['arena_radius'], sim_options['max_time'], num_threads)
    elif strategy == "EBN":
        performanceEvaluationForMultiEBNs(sim_options['num_robots'], num_nodes, sim_options['arena_radius'], sim_options['max_time'], num_threads)
    # performanceEvaluationForEBN(num_robots, num_nodes, arena_radius)

    print("Time running: %s" % (time.time() - start_time))