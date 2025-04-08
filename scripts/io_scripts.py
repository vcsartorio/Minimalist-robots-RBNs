import csv
import os
import numpy as np
import pandas as pd
import math
from scipy import stats
import scripts.utils as utils
import src.utils.BooleanNetwork as BooleanNetwork
from operator import itemgetter
from lxml import etree

### --------------------------------- Input Scripts ------------------------------------ ###

def readSimulationConfigFile():
    sim_config_options = "simulation_config/sim_config.xml"
    sim_config = dict()

    try:
        tree = etree.parse(sim_config_options)
        root = tree.getroot()
        sim_config['max_time'] = int(root.get("max_time"))
        sim_config['arena_radius'] = float(root.get("arena_radius"))
        sim_config['num_robots'] = int(root.get("num_robots"))
    except Exception as e:
        print("Couldnt read config simulation options!\n" + str(e))

    return sim_config

def readEBNsfromEvolutionFolder(evolution_main_dir, num_robots, num_nodes, date, target_type):
    all_data = []
    all_dirs = os.listdir(evolution_main_dir)
    all_dirs.sort()
    for evolution_dir in all_dirs:
        algorithm_name = ""
        file_robots = "-1"
        file_nodes = "-1"
        file_date = ""
        file_targets = ""
        elements = evolution_dir.split("_")
        algorithm_name = elements[1]
        for e in elements:
            if (e.startswith("202")):
                file_date = e
            if e.endswith("R"):
                file_robots = int(e[:-1])
            if (e.endswith("N")):
                file_nodes = int(e[:-1])
            if (e.endswith("targets")):
                file_targets = e

        if (algorithm_name != "ga"):
            continue

        # Check specific number of node and robots
        if (file_robots != num_robots or file_nodes != num_nodes):
            continue

        # Check date of experiments
        if ((not file_date.startswith(date))):
            continue

        if (file_targets != target_type):
            continue

        print(evolution_dir)
        disc_weibull = []
        disc_time = []
        frac_disc = []
        inf_time = []
        frac_inf = []
        evolution_curve_path_txt = evolution_main_dir + "/" + evolution_dir + "/evolution_curve_data.txt"
        if os.path.exists(evolution_curve_path_txt):
            with open(evolution_curve_path_txt, 'r') as data:
                try:
                    lines = data.readlines()
                    for line in lines:
                        disc_weibull.append(int(line.split()[0])/32.0)
                        disc_time.append(int(line.split()[1]))
                        frac_disc.append(float(line.split()[2]))
                        inf_time.append(int(line.split()[3]))
                        frac_inf.append(float(line.split()[4]))
                    data.close()
                except Exception as e:
                    print("Couldnt open data file!\n" + str(e))
        else:
            with open(evolution_main_dir + "/" + evolution_dir + "/evolution_curve_data.tsv", 'r') as data:
                try:
                    read_tsv = csv.reader(data, delimiter="\t")
                    next(read_tsv, None)
                    for row in read_tsv:
                        disc_weibull.append(float(row[1])/32.0)
                        disc_time.append(float(row[2])/32.0)
                        frac_disc.append(float(row[3]))
                        inf_time.append(float(row[4])/32.0)
                        frac_inf.append(float(row[5]))
                    data.close()
                except Exception as e:
                    print("Couldnt open data file!\n" + str(e))

        all_data.append(dict())
        all_data[-1]["dir_name"] = evolution_dir
        all_data[-1]["weibull_disc"] = disc_weibull
        all_data[-1]["disc_time"] = disc_time
        all_data[-1]["inf_time"] = inf_time
        all_data[-1]["frac_disc"] = frac_disc
        all_data[-1]["frac_inf"] = frac_inf

    return all_data

def getEBNsParametersFileFromPostEvaFolder(posteva_folder, num_node, num_robots, evolution_targets_type):
    ebn_parameters = []
    for subdir, dirs, files in os.walk(posteva_folder):
        dirs.sort()
        for folder in dirs:
            elements = folder.split("_")
            date = ""
            file_node = -1
            file_robots = -1
            file_seed = -1
            file_targets = ""
            for e in elements:
                if e.endswith("N"):
                    file_node = int(e.replace("N", ""))
                if e.endswith("R"):
                    file_robots = int(e.replace("R", ""))
                if e.startswith("202"):
                    date = e  
                if e.endswith("seed"):
                    file_seed = int(e.replace("seed", ""))
                if e.endswith("targets"):
                    file_targets = e             

            if file_node != num_node or num_robots != file_robots:
                continue

            if evolution_targets_type != file_targets:
                continue

            if file_seed in [3,5,6,7] and file_node == 18:
                continue

            # if not date.startswith(experiment_date):
            #     continue
            # if date != "2023-09-18" and date != "2023-10-04" and date != "2023-10-06":
            #     continue

            # ebn_parameters[file_nodes]["date"] = date
            ebn_parameters.append(dict())
            ebn_parameters[-1]["node"] = file_node
            ebn_parameters[-1]["seed"] = file_seed
            ebn_parameters[-1]["date"] = date
            print(f'Reading folder: {folder}')
            for file_name in os.listdir(posteva_folder + folder):
                if file_name == "initial-states.txt":
                    ebn_parameters[-1]["initial state"] = posteva_folder + folder + "/" + file_name
                if file_name == "parameters_best_ebn.txt":
                    ebn_parameters[-1]["parameters"] = posteva_folder + folder + "/" + file_name

    return ebn_parameters

def readEBNsTsvFptPerformance(fpt_folder, num_node, num_robots, evolution_targets_type, ebn_parameters):
    for file_name in os.listdir(fpt_folder):
        elements = file_name.replace(".tsv", "").split("_")
        network_type = elements[0]
        file_node = -1
        file_robots = -1
        file_id = ""
        for e in elements:
            if e.endswith("N"):
                file_node = int(e.replace("N", ""))
            if e.endswith("R"):
                file_robots = int(e.replace("R", ""))
            if e.startswith("id"):
                file_id = e.replace("id:", "")

        if network_type != "ebn" or file_node != num_node or num_robots != file_robots:
            continue
        
        if file_id != evolution_targets_type:
            continue
        
        print(f'Reading file: {file_name}')
        with open(fpt_folder + file_name, "r") as dfile:
            try:
                read_tsv = csv.reader(dfile, delimiter="\t")
                next(read_tsv, None)
                for row in read_tsv:
                    seed = int(row[0])
                    for ebn in ebn_parameters:
                        if ebn["node"] == file_node and ebn["seed"] == seed:
                            if "fpt" not in ebn.keys():
                                ebn["fpt"] = [float(row[1])/32.0]
                            else:
                                ebn["fpt"].append(float(row[1])/32.0)
                    
                dfile.close()
            except Exception as e:
                print(("Couldnt read results file: %s!\nException: " + str(e)) % (file_name))

    print(ebn_parameters)
    for idx, ebn in enumerate(ebn_parameters):
        ebn_parameters[idx]["fpt"] = sum(ebn["fpt"]) / len(ebn["fpt"])

    return ebn_parameters

def readEBNsParameters(num_nodes, num_robots, main_folder, experiment_date):
    parameters_files = []
    init_state_files = []
    for subdir, dirs, files in os.walk(main_folder):
        dirs.sort()
        for folder in dirs:
            elements = folder.split("_")
            date = ""
            file_nodes = -1
            file_robots = -1
            for e in elements:
                if e.endswith("N"):
                    file_nodes = int(e.replace("N", ""))
                if e.endswith("R"):
                    file_robots = int(e.replace("R", ""))
                if e.startswith("2023"):
                    date = e  

            if file_nodes not in num_nodes or file_robots != num_robots:
                continue

            # if not date.startswith(experiment_date):
            #     continue

            print(f'Reading folder: {folder}')
            for file_name in os.listdir(main_folder + folder):
                if file_name == "initial-states.txt":
                    init_state_files.append(main_folder + folder + "/" + file_name)
                if file_name == "parameters_best_ebn.txt":
                    parameters_files.append(main_folder + folder + "/" + file_name)

    parameters_files.sort()
    init_state_files.sort()

    return parameters_files, init_state_files

def readEBNsFromFile(ebn_parameters_folder, num_nodes, num_robots):
    parameters_files = []
    for file_name in os.listdir(ebn_parameters_folder):
        elements = file_name.replace(".txt", "").split("_")
        file_type = elements[0]
        date = ""
        file_nodes = -1
        file_robots = -1
        for e in elements:
            if e.endswith("N"):
                file_nodes = int(e.replace("N", ""))
            if e.endswith("R"):
                file_robots = int(e.replace("R", ""))
            if e.startswith("202"):
                date = e  

        if file_nodes not in num_nodes or num_robots != file_robots:
            continue

        if file_type == "initial":
            continue
        else:
            parameters_files.append(file_name)

    parameters_files.sort()

    ebns = []
    for idx, parameters_file_name in enumerate(parameters_files):
        ebn = BooleanNetwork.BooleanNetwork(file_nodes, read_from=(ebn_parameters_folder + parameters_file_name))
        ebns.append(ebn)

    return ebns

def readRBNsParametersAndPerformance(network_performance_file, num_nodes):
    all_rbns = []
    all_parameters = []
    weibull = 0
    rbn_file = network_performance_file
    rbn_file += "RBN_20R_%dN_parameters_and_fpt.txt" % (num_nodes)
    # rbn_file += "EBN_20R_%dN_parameters_and_fpt.txt" % (num_nodes)
    with open(rbn_file, "r+") as file:
        try:
            lines = file.readlines()
            for line in lines:
                if line == "\n":
                    new_rbn = BooleanNetwork.BooleanNetwork(num_nodes, bn_type="RBN", parameters=all_parameters)
                    new_rbn.setPerformanceExperiment(20, 100, 0.475, True, fpt=weibull)
                    all_rbns.append(new_rbn)
                    all_parameters = []
                elif "fpt:" not in line:
                    if len(all_parameters) < ((num_nodes*num_nodes) + (num_nodes*(num_nodes-1))):
                        all_parameters += list(map(int, line.strip('\n')))
                elif "fpt:" in line:
                    weibull = int(line.split()[1])
                else:
                    continue
            file.close()
        except Exception as e:
            print("Couldnt open RBNs file. Error: " + str(e))

    # all_rbns = sorted(all_rbns, key=operator.attrgetter('weibull_discovery_time'))

    return all_rbns

def readResultsFromFolder(folder, strategies, num_robots, nodes, alpha, rho, trials):
    experiments_dict = utils.createExperimentsDict()
    for subdir, dirs, files in os.walk(folder):
        files.sort()

        for file_name in files:
            experiment_parameters = utils.getFileParameters(file_name)
            if not utils.selectExperiment(experiment_parameters, strategies, num_robots, nodes, alpha, rho, trials):
                continue

            # with open(data_folder + data_file, "r") as dfile:
            #     try:
            #         label, order = labelName.setLabelName(data_file.replace(".tsv", ""))
            #         name = label.replace(" No Bias", "")
            #         read_tsv = csv.reader(dfile, delimiter="\t")
            #         next(read_tsv, None)
            #         data_dict[name] = []
            #         for row in read_tsv:
            #             data_dict[name].append(float(row[1])/32.0)
            #         dfile.close()
            #     except Exception as e:
            #         print(("Couldnt read results file: %s!\n" + str(e)) % (data_file))

            print(f"Loading: {file_name}")
            experiments_dict = openTSVtoList(experiments_dict, folder, file_name, experiment_parameters)

    df = pd.DataFrame(experiments_dict)
    return df

def openTSVtoList(experiments_dict, folder, file_name, experiment_parameters):
    # print(f'File name: {file_name}')
    with open(folder + file_name, "r") as dfile:
        try:
            label = utils.setLabelName(file_name.replace(".tsv", ""))
            label_elements = label.split(" ")
            read_tsv = csv.reader(dfile, delimiter="\t")
            next(read_tsv, None)
            for row in read_tsv:
                if experiment_parameters['num_net'] > 1:
                    # experiments_dict['Name'].append(f"{label} {int(row[0])+1}")
                    # experiments_dict['Name'].append(f"{label}")
                    # experiments_dict['Strategy'].append(label_elements[0])
                                        
                    if label_elements[0] == "EBN":
                        # experiments_dict['Name'].append(f"{label_elements[1]}")
                        experiments_dict['Name'].append(f"{label_elements[1]}-{int(row[0])}")
                        # label_aux = label_elements[1].replace("N", "")
                        # experiments_dict['Name'].append(f"{label_aux}-{int(row[0])+1}")
                        # experiments_dict['Strategy'].append(label_elements)
                        # if experiment_parameters['id'] != "":
                        #     experiments_dict['Name'][-1] += " " + experiment_parameters['id']
                        #     experiments_dict['Strategy'][-1] += " " +  experiment_parameters['id']
                        experiments_dict['Strategy'].append(label_elements[0])
                    elif label_elements[0] == "RBN":
                        # label_aux = label_elements[1].replace("N", "")
                        # experiments_dict['Name'].append(f"{label_aux}")
                        # experiments_dict['Name'].append(f"{label}-{int(row[0])+1}")
                        # experiments_dict['Name'].append(f"{label_elements[1]}-{int(row[0])+1}")
                        # experiments_dict['Name'].append(f"{label}")
                        experiments_dict['Name'].append(f"{label_elements[1]}")
                        experiments_dict['Strategy'].append(label_elements[0])
                        # experiments_dict['Strategy'].append(label)
                    else:
                        experiments_dict['Name'].append(f"{label}")
                        experiments_dict['Strategy'].append(label_elements[0])
                else:
                    # experiments_dict['Name'].append(f"LC")
                    experiments_dict['Name'].append(label_elements[0])
                    experiments_dict['Strategy'].append(label_elements[0])

                experiments_dict['Parameters'].append(label_elements[1])
                if experiment_parameters['bias']:
                    experiments_dict['Bias'].append("Bias")
                else:
                    experiments_dict['Bias'].append("No Bias")

                experiments_dict['First Passage Time'].append(float(row[1])/32.0)
                experiments_dict['Fraction Discovery'].append(float(row[3]))
                experiments_dict['Robots'].append(experiment_parameters['robots'])
                experiments_dict['Encode'].append(experiment_parameters['encode'])
                experiments_dict['Arena Size'].append(experiment_parameters['arena'])
                experiments_dict['id'].append(experiment_parameters['id'])

            dfile.close()
        except Exception as e:
            print(("Couldnt read results file: %s!\nException: " + str(e)) % (file_name))

    return experiments_dict

def loadPositionsFile(dir_path, pck_filename):
    num_experiment = len([name for name in os.listdir(dir_path) if
                          (os.path.isfile(os.path.join(dir_path, name)) and (name.endswith('position.tsv')))])
    
    if os.path.exists(dir_path + "/" + pck_filename + ".pkl"):
        return num_experiment, pd.read_pickle(dir_path + "/" + pck_filename + ".pkl")
    
    return num_experiment, createPickleFile(dir_path, pck_filename)

def createPickleFile(dir_path, file_name):
    print("Generating pickle positions file in " + dir_path + "/" + file_name + ".pkl")
    df = pd.DataFrame()
    for pos_filename in os.listdir(dir_path):
        if pos_filename.endswith('position.tsv'):
            if not os.path.getsize(os.path.join(dir_path, pos_filename)) > 0:
                print("Error, empty file at:" + os.path.join(dir_path, pos_filename))
                continue
            df_single = pd.read_csv(dir_path + "/" + pos_filename, sep="\t")
            df = df.append(df_single)

    df.to_pickle(dir_path + "/" + file_name + ".pkl")
    return df

def readChaoticBehaviourFile(save_chaos_test_path, all_rbns, trials, evaluations, max_steps):
    fpt = []
    delta = []
    comp = []
    entropy = []
    diseq = []
    sl = []
    node = 0
    bn_type = ""
    read_file = save_chaos_test_path + "behaviour-results_%dk_%de_%dt.txt" % (trials, evaluations, max_steps)
    with open(read_file, "r") as save_file:
        try:
            lines = save_file.readlines()
            for line in lines:
                if line == "\n":
                    print("%s %d" % (bn_type, node))
                    label = "%s %d" % (bn_type, node)
                    if label not in all_rbns:
                        all_rbns[label] = dict()
                    all_rbns[label] = dict()
                    all_rbns[label]["fpt"] = list(fpt)
                    all_rbns[label]["delta"] = list(delta)
                    all_rbns[label]["complexity"] = list(comp)
                    all_rbns[label]["entropy"] = list(entropy)
                    all_rbns[label]["disequilibrium"] = list(diseq)
                    all_rbns[label]["sl"] = list(sl)
                    fpt.clear()
                    delta.clear()
                    entropy.clear()
                    diseq.clear()
                    comp.clear()
                    sl.clear()
                    bn_type = ""
                elif "BN" in line.split()[0]:
                    bn_type = line.split()[0]
                    node = int(line.split()[1].replace("\n", ""))
                else:
                    fpt.append(int(line.split()[0]))
                    delta.append(float(line.split()[1]))
                    comp.append(float(line.split()[2]))
                    entropy.append(float(line.split()[3]))
                    diseq.append(float(line.split()[4]))
                    sl.append(float(line.split()[5]))
            save_file.close()
        except Exception as e:
            print(f"Couldnt read {read_file} PerformancexChaos file. Error: {e}")

    return all_rbns

### ------------------------------------- Output Scripts -------------------------------- ###

def savePerformanceOverChaos(save_chaos_test_path, rbn, node, bn_type, max_steps, evaluations):
    # for node in all_nodes:
    #     np.array([(1,2),(3,4)], dtype="f,f")
    # df_experiment = pd.DataFrame(data=all_rbns, index=all_nodes)
    # print(df_experiment)
    # [number_of_experiments, df_experiment] = utils.load_pd_positions(dirName, "experiment")
    # positions_concatenated = df_experiment.values[:, 1:1800]
    # [num_robot, num_times] = positions_concatenated.shape
    # positions_concatenated = np.array([x.split(',') for x in positions_concatenated.ravel()], dtype=float)
    # positions_concatenated = positions_concatenated.reshape(num_robot, num_times, 2)
    # pos_transposed = positions_concatenated.transpose(1,0,2)
    # # print(pos_transposed)
    # if num_nodes == "-1":
    #     np.save(np_folder +'/positions_CRWLEVY_robots#20_alpha#%s_rho#%s.npy' % (alpha, rho), pos_transposed)

    print(f"Saving behaviour results on file for {bn_type} {node}N")
    trials = len(rbn["fpt"])
    encode = ""
    with open(save_chaos_test_path + "behaviour-results_%dk_%de_%dt.txt" % (trials, evaluations, max_steps), "a+") as save_file:
        try:
            save_file.write("%s %d%s\n" % (bn_type, node, " encoded" if encode == "encoded" else ""))
            for i in range(trials):
                save_file.write("%d %.4f %.2f %.2f %.2f %d\n" % (rbn["fpt"][i], 
                    rbn["delta"][i], rbn["complexity"][i], rbn["entropy"][i], rbn["disequilibrium"][i], rbn["sl"][i]))
            save_file.write("\n")
            save_file.close()
        except Exception as e:
            print("Couldnt save PerformancexChaos file. Error: " + str(e))