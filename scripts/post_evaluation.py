import random
import os
import numpy as np
import src.utils.BooleanNetwork as BooleanNetwork
import src.utils.Targets as Targets
import src.KilobotsSearchExperiment as KilobotsSearchExperiment

KILOBOT_BIAS = True

def evolutionParameters(folder):
    num_nodes = -1
    num_robots = -1
    date = ""
    seed = 0
    target_type = ""

    folder_name = folder.split("/")[-1]
    elements = folder_name.split("_")
    for e in elements:
        if e.startswith("202"):
            date = e
        if e.endswith("N"):
            num_nodes = int(e.replace("N", ""))
        if e.endswith("R"):
            num_robots = int(e.replace("R", ""))
        if e.endswith("seed"):
            seed = int(e.replace("seed", ""))
        if e.endswith("targets"):
            target_type = e

    return num_nodes, num_robots, date, seed, target_type

def readEBNsParameters(num_nodes, num_robots, max_trials, arena_radius, bias, folder):
    ebns = []
    connections = []
    bfunctions = []
    generation = -1
    fpt = -1
    with open(folder + "/best_individuals_data.txt", "r") as file:
        try:
            lines = file.readlines()
            for line in lines:
                if line == "\n":
                    network_parameters = np.array(connections + bfunctions)
                    network_parameters.flatten()
                    new_ebn = BooleanNetwork.BooleanNetwork(num_nodes, bn_type="EBN", parameters=network_parameters, net_id=f'{generation:04}')
                    new_ebn.setPerformanceExperiment(num_robots, max_trials, arena_radius, bias, save_exp=False, fpt=fpt)
                    ebns.append(new_ebn)
                    connections.clear()
                    bfunctions.clear()
                elif "fitness:" not in line:
                    if len(line) == (num_nodes + 1):
                        connections += list(map(int, line.strip('\n')))
                    else:
                        bfunctions += list(map(int, line.strip('\n')))
                elif "fitness:" in line:
                    generation = int(line.split()[-1])
                    fpt = float(line.split()[1])

            file.close()
        except Exception as e:
            print("Couldnt open best individuals file. Error: " + str(e))

    return ebns

def selectBestNEBNs(ebns, num_net):
    new_ebns = [ebns[0]]
    worst_ebn = {'fpt': ebns[0].fpt_result, 'idx': 0}
    for ebn in ebns:
        if len(new_ebns) < num_net:
            new_ebns.append(ebn)
            if ebn.fpt_result > worst_ebn['fpt']:
                worst_ebn['fpt'] = ebn.fpt_result
                worst_ebn['idx'] = len(new_ebns)-1
        else:
            if ebn.fpt_result < worst_ebn['fpt']:
                del new_ebns[worst_ebn['idx']]
                new_ebns.append(ebn)
                worst_ebn['fpt'], worst_ebn['idx'] = max([(e.fpt_result, idx) for idx, e in enumerate(new_ebns)], key=lambda item:item[0])      

    return new_ebns

def readInitialStates(folder):
    initial_states = []
    with open(folder + "/initial_state_file.txt", "r+") as init_file:
        try:
            lines = init_file.readlines()
            for line in lines:
                initial_states.append([])
                for ch in line:
                    if ch != '\n':
                        initial_states[-1].append(int(ch))
            init_file.close()
        except Exception as e:
            print("Couldnt open initial states file. Error: " + str(e))

    return initial_states

def createSaveFolder(num_nodes, num_robots, date, seed, target_type):
    save_folder = "/input/" + f"/ga_{num_robots}R_{num_nodes}N_{date}_{target_type}_{seed}seed"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return save_folder

def savePostEvaluatuion(ebns, num_nodes, save_folder):
    with open(save_folder + f"/posteva_best_ebns.txt", "a+") as file:
        try:
            for ebn in ebns:
                file.write(f"fitness: {int(ebn.experiment_performance.weibull_discovery_time)} {int(ebn.experiment_performance.discovery_time)} {ebn.experiment_performance.fraction_discovery:.2f} {int(ebn.experiment_performance.information_time)} {ebn.experiment_performance.fraction_information:.2f} Gen: {int(ebn.net_id)}\n")
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        file.write("%d" % ebn.connections[i][j])
                    file.write("\n")
                for i in range(num_nodes):
                    for j in range(ebn.num_bfunction):
                        file.write("%d" % ebn.bfunctions[i][j])
                    file.write("\n")
                file.write("\n")

            file.close()
        except Exception as e:
            print("Couldnt save post evaluation file. Error: " + str(e))

def saveParametersFromBestEBNAfterPosteva(ebns, save_folder):
    # best_ebn = min([int(ebn.experiment_performance.weibull_discovery_time) for ebn in ebns])
    best_ebn = sorted(ebns, key=lambda x: x.experiment_performance.weibull_discovery_time)[0]
    with open(save_folder + f"/parameters_best_ebn.txt", "a+") as file:
        try:
            for i in range(best_ebn.num_nodes):
                for j in range(best_ebn.num_nodes):
                    file.write("%d" % best_ebn.connections[i][j])
                file.write("\n")
            for i in range(best_ebn.num_nodes):
                for j in range(best_ebn.num_bfunction):
                    file.write("%d" % best_ebn.bfunctions[i][j])
                file.write("\n")
            file.close()
        except Exception as e:
            print("Couldnt save best ebn file. Error: " + str(e))

def saveInitialStates(initial_states, save_folder):
    with open(save_folder + "/initial-states.txt", "w+") as init_file:
        try:
            init_file.truncate(0)
            for i in initial_states:
                for j in i:
                    init_file.write(str(j))
                init_file.write("\n")
            init_file.close()
        except Exception as e:
            print("Error! Couldnt save initial state:" + str(e))


def runPostEvaluation(num_threads):
    seed_folders = ["/input/Evolutions/2024-05-22_ga_20R_28N_random-targets_6seed"]
    
    max_trials = 100
    arena_radius = 0.475
    simulation_time = 3000
    num_net_for_posteva = 25
    print("Post-evaluation starting for %d seeds." % (len(seed_folders)))
    for evolution_folder in seed_folders:
        num_nodes, num_robots, date, seed, target_type = evolutionParameters(evolution_folder)
        save_folder = createSaveFolder(num_nodes, num_robots, date, seed, target_type)

        ebns = readEBNsParameters(num_nodes, num_robots, max_trials, arena_radius, KILOBOT_BIAS, evolution_folder)
        ebns = selectBestNEBNs(ebns, num_net_for_posteva)
        initial_states = readInitialStates(evolution_folder)
        saveInitialStates(initial_states, save_folder)
        for ebn in ebns:
            ebn.setRobotsInitialState(initial_states)

        random.seed(15)
        targets_position = Targets.createTargetPosition(max_trials, False, arena_radius)

        print("Starting post-evaluation for %dN EBN seed:%s" % (num_nodes, date))
        experiment = KilobotsSearchExperiment.NetworkKilobotsExperiment(num_threads, num_robots, targets_position, arena_radius, simulation_time, KILOBOT_BIAS, save_pos=False, save_parameters=False)
        experiment.executeKilobotExperimentTrials(ebns)

        savePostEvaluatuion(ebns, num_nodes, save_folder)
        saveParametersFromBestEBNAfterPosteva(ebns, save_folder)