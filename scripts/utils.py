import csv
import os
import numpy as np
import pandas as pd
import math
import random
from scipy import stats
import scripts.io_scripts as io_scripts
import src.utils.BooleanNetwork as BooleanNetwork

def sortXLabelPlot(name_list):
    copy_list = []
    ebn_names = []
    for name in name_list:
        if name.startswith("EBN"):
            ebn_names.append(name)
        else:
            copy_list.append(name)

    return copy_list + ebn_names

def selectExperiment(experiment_parameters, strategies, num_robots, nodes, alpha, rho, trials):
    if experiment_parameters['strategy'] not in strategies:
        return False
    if experiment_parameters['robots'] != num_robots:
        return False
    if experiment_parameters['bias'] == False:
        return False
    if experiment_parameters['encode'] != "direct":
        return False
    if experiment_parameters['trials'] != trials:
        return False
    if experiment_parameters['evaluations'] != 20:
        return False

    if experiment_parameters['strategy'] == "crwlevy":
        if experiment_parameters['alpha'] != alpha:
            return False
        if experiment_parameters['rho'] != rho:
            return False
    else:
        if experiment_parameters['nodes'] not in nodes:
        # if experiment_parameters['nodes'] not in nodes and experiment_parameters['strategy'] == "rbn":
            return False
        if experiment_parameters['evaluations'] != 20:
            return False
        if experiment_parameters['strategy'] == "ebn":
            if experiment_parameters['num_net'] < 2:
                return False
            if experiment_parameters['id'] == "fixed-targets":
                return False
        else:
            if experiment_parameters['num_net'] != 100:
                return False

    return True

def setLabelName(file_name):
    label = ""
    name_elements = file_name.split("_")
    if name_elements[0] == "crwlevy":
        #label += "LMCRW " + r'$\alpha$:' + name_elements[2].replace("a", "") + r' $\rho$:' + name_elements[3].replace("p", "")
        label += f'LMCRW a:{name_elements[2].replace("a", "")}p:{name_elements[3].replace("p", "")}'
    elif name_elements[0] == "crwrandom":
        #label += "CRWRANDOM " + r'$step range:' + name_elements[2].replace("steps:", "") + r' $\rho$:' + name_elements[3].replace("p", "")
        label += f'CRWRANDOM sr:{name_elements[2].replace("steps:", "")}p:{name_elements[3].replace("p", "")}'
    elif name_elements[0] == "rbn":
        label += f'RBN {name_elements[2]}'
    elif name_elements[0] == "ebn":
        label += f'EBN {name_elements[2]}'
    else:
        print("Label script: couldnt find experiment type.")
        
    return label

def createExperimentsDict():
    experiments_dict = dict()
    experiments_dict['Name'] = []
    experiments_dict['Strategy'] = []
    experiments_dict['Parameters'] = []
    experiments_dict['Bias'] = []
    experiments_dict['First Passage Time'] = []
    experiments_dict['Fraction Discovery'] = []
    experiments_dict['Robots'] = []
    experiments_dict['Encode'] = []
    experiments_dict['Arena Size'] = []
    experiments_dict['id'] = []

    return experiments_dict

def getFileParameters(file_name):
    experiment_parameters = dict()
    experiment_parameters['strategy'] = ""
    experiment_parameters['nodes'] = -1
    experiment_parameters['alpha'] = -1
    experiment_parameters['rho'] = -1
    experiment_parameters['robots'] = -1
    experiment_parameters['trials'] = 100
    experiment_parameters['num_net'] = 1
    experiment_parameters['evaluations'] = 1
    experiment_parameters['arena'] = 90
    experiment_parameters['bias'] = True
    experiment_parameters['encode'] = "direct"
    experiment_parameters['id'] = ""

    elements = file_name[:-4].split("_")
    experiment_parameters['strategy'] = elements[0]
    for e in elements[1:]:
        if e.endswith("R"):
            experiment_parameters['robots'] = int(e.replace("R", ""))
        if e.endswith("N"):
            experiment_parameters['nodes'] = int(e.replace("N", ""))
        if e.endswith("a"):
            experiment_parameters['alpha'] = eval(e.replace("a", ""))
        if e.endswith("p"):
            experiment_parameters['rho'] = eval(e.replace("p", ""))
        if e.endswith("k"):
            experiment_parameters['trials'] = int(e.replace("k", ""))
        if e.endswith("n"):
            experiment_parameters['num_net'] = int(e.replace("n", ""))
        if e.endswith("e"):
            experiment_parameters['evaluations'] = int(e.replace("e", ""))
        if e.endswith("cm"):
            experiment_parameters['arena'] = int(e.replace("cm", ""))
        if e.startswith("id:"):
            experiment_parameters['id'] = e.replace("id:", "")
        if e == "NoBias":
            experiment_parameters['bias'] = False
        if e.endswith("encoded"):
            experiment_parameters['encode'] = e

    return experiment_parameters

def selectBestResultsFromDF(df):
    rbn_fpt_values = dict()
    lmcrw_fpt_values = []
    for result in df['Name'].unique():
        if "LMCRW" in result:
            lmcrw_fpt_values = df.loc[df['Name'] == result]["First Passage Time"].values
        else:
            rbn_fpt_values[result] = dict()
            rbn_fpt_values[result]["First Passage Time"] = df.loc[df['Name'] == result]["First Passage Time"].values

    lmcrw_mean_fpt = np.mean(lmcrw_fpt_values)
    # print(f"lmcrw mean fpt: {lmcrw_mean_fpt}")
    for rbn in rbn_fpt_values:
        rbn_fpt_values[rbn]['mean'] = np.mean(rbn_fpt_values[rbn]["First Passage Time"])
        # print(f"{rbn} mean fpt: {rbn_fpt_values[rbn]['mean']}")

    best_results = [name for name in rbn_fpt_values.keys() if rbn_fpt_values[name]['mean'] <= lmcrw_mean_fpt]
    best_results.append('LMCRW')
    df['Show'] = pd.Series([False]).bool()
    for results in best_results:
        df['Show'] = pd.Series(np.where(df['Name'] == results, True, df['Show']), dtype='boolean')

    df.drop(df[(df['Show'] == False)].index, inplace=True)
    print(df)
    return df

def selectNfromBestResults(df, n_samples):
    rbn_fpt_values = []
    lmcrw_fpt_values = []
    ebn_names = []
    for result in df['Name'].unique():
        if "LMCRW" in result:
            lmcrw_fpt_values = df.loc[df['Name'] == result]["First Passage Time"].values
        elif "RBN" in result:
            rbn_fpt_values.append(dict())
            rbn_fpt_values[-1]["Name"] = result
            rbn_fpt_values[-1]["First Passage Time"] = df.loc[df['Name'] == result]["First Passage Time"].values
        else:
            ebn_names.append(result)

    lmcrw_mean_fpt = np.mean(lmcrw_fpt_values)
    # print(f"lmcrw mean fpt: {lmcrw_mean_fpt}")
    for rbn in rbn_fpt_values:
        rbn["mean"] = np.mean(rbn["First Passage Time"])
        # print(f"{rbn['Name']} mean fpt: {rbn['mean']}")

    rbn_fpt_values = sorted(rbn_fpt_values, key=lambda d: d['mean']) 
    best_results = [results["Name"] for results in rbn_fpt_values[:n_samples]]
    best_results.append('LMCRW')
    best_results = best_results + ebn_names
    df['Show'] = pd.Series([False]).bool()
    for results in best_results:
        df['Show'] = pd.Series(np.where(df['Name'] == results, True, df['Show']), dtype='boolean')

    df.drop(df[(df['Show'] == False)].index, inplace=True)
    # print(df)
    return df

def sampleRBNCondition(name, number):
    if name.startswith("RBN") and name.split("-")[-1].endswith(str(number)):
        return 5
    return 42

def selectNRandomResults(df, n_samples):
    random_nets  = random.sample(range(0,101), 101-n_samples)
    df['Show'] = True
    # print(df)
    for number in random_nets:
        df['Show'] = pd.Series(np.where((df['Name'].str.startswith("RBN")) & (df['Name'].str.endswith("-" + str(number))), False, df['Show']), dtype='boolean')

    df.drop(df[(df['Show'] == False)].index, inplace=True)
    return df

def getResultsNameBetterThan(df, comparison_name):
    rbn_fpt_values = dict()
    lmcrw_fpt_values = []
    for result in df['Name'].unique():
        if comparison_name in result:
            lmcrw_fpt_values = df.loc[df['Name'] == result]["First Passage Time"].values
        else:
            rbn_fpt_values[result] = dict()
            rbn_fpt_values[result]["First Passage Time"] = df.loc[df['Name'] == result]["First Passage Time"].values

    lmcrw_mean_fpt = np.mean(lmcrw_fpt_values)
    print(f"Comparing with {comparison_name}. Mean fpt: {lmcrw_mean_fpt}")
    for rbn in rbn_fpt_values:
        rbn_fpt_values[rbn]['mean'] = np.mean(rbn_fpt_values[rbn]["First Passage Time"])
        if rbn_fpt_values[rbn]['mean'] < lmcrw_mean_fpt:
            print(f"{rbn} mean fpt: {rbn_fpt_values[rbn]['mean']}")

    best_results = [name for name in rbn_fpt_values.keys() if rbn_fpt_values[name]['mean'] <= lmcrw_mean_fpt]
    return best_results

def setComparisonWith(df, comparison_name, dict_results):
    df['Comparison'] = comparison_name
    for name in dict_results.keys():
        strategy = name.split()[:2]
        strategy = " ".join(strategy)
        result = f'{strategy} {dict_results[name]}'
        # df['Comparison'] = np.where(df['Name'] == name, result, df['Comparison'])
        df['Comparison'] = np.where(df['Name'] == name, dict_results[name], df['Comparison'])

    return df

def createStrategyComparisonDF(df):
    print(df)
    new_dict = {'Name': [], 'Value': [], 'Comparison': []}
    strategies = df['Strategy'].unique()
    for strategy in strategies:
        print(strategy)
        rows = df.loc[df['Strategy'] == strategy]
        result = rows['Comparison'].value_counts(normalize=True)
        print(result)
        for comparison in ["Worst", "Equal", "Better"]:
            new_dict["Name"].append(strategy)
            if comparison in result.index:
                new_dict["Value"].append(result[comparison])
            else:
                new_dict["Value"].append(0.0)
            new_dict["Comparison"].append(comparison)

    new_df = pd.DataFrame(data=new_dict)
    return new_df

def deleteValuesOutOfArena(grid, radius):
    index = np.arange(-0.475, 0.476, radius, dtype=float)
    cont = 0
    for i1, x in enumerate(index):
        for i2, y in enumerate(index):
            radius = math.sqrt(pow(x,2) + pow(y,2))
            if radius > 0.475:
                grid.iloc[i1,i2] = None
                cont += 1
    return grid

def createArenaGrid(radius):
    index = np.arange(-0.475, 0.476, radius, dtype=float)
    index_str = ['{:.3f}'.format(x) for x in index]
    d = dict()
    for i in index_str:
        d[i] = np.zeros(len(index), dtype=int)

    grid = pd.DataFrame(data=d, index=index_str)
    return deleteValuesOutOfArena(grid, radius)

def addPositionsToArenaGrid(grid, pos_x, pos_y, radius):
    index = np.arange(-0.475, 0.476, radius, dtype=float)
    for i in range(len(pos_x)):
        col_grid = '{:.3f}'.format(min(index, key=lambda x:abs(x-pos_y[i])))
        idx_grid = '{:.3f}'.format(min(index, key=lambda x:abs(x-pos_x[i])))
        # print("%f %f -> %s %s" % (pos_x[i], pos_y[i], col, idx))
        grid.loc[col_grid, idx_grid] += 1

    return grid

def standarization(grid, radius):
    # mean = all_grids[count_grid].mean()
    # std = all_grids[count_grid].std()
    # all_grids[count_grid] = (all_grids[count_grid]-mean)/std
    # all_grids[count_grid] = all_grids[count_grid].replace(np.nan,0)
    # index = np.arange(-0.475, 0.476, kilobot_comm_radius, dtype=float)
    # all_grids[count_grid] = utils.deleteValuesOutOfArena(all_grids[count_grid], index)
    # all_grids[count_grid] = all_grids[count_grid].apply(lambda value: (abs(value-mean)/std), axis=1)
    # all_grids[count_grid] = [utils.standarization(row, mean, std) for row in all_grids[count_grid][:].to_numpy()]
    # all_grids[count_grid] = stats.zscore(all_grids[count_grid], nan_policy='omit')
    # df_values = all_grids[count_grid].values
    # print(df_values.shape)
    # df_values = stats.zscore(df_values, nan_policy='omit')
    # new_df = pd.DataFrame(df_values.reshape(all_grids[count_grid].shape[0],all_grids[count_grid].shape[1]))
    # print(new_df)
    # pd.melt(stacked)
    # print(df_stacked.shape)
    # df_stacked.iloc[:,-1:].apply(stats.zscore)
    # print(df_stacked)
    grid = grid.fillna(0)
    stacked = grid.stack()
    stacked_stand = stats.zscore(stacked, nan_policy='omit')
    df_unstack = stacked_stand.unstack(fill_value=0)
    return deleteValuesOutOfArena(df_unstack, radius)

def transformTimeSimulationToPerc(df):
    sizes = [45,90,180]
    sim_time = [750,3000,12000]
    for count, size in enumerate(sizes):
        # df.loc[df['Arena Size'] == size]["First Passage Time"].apply(lambda value: value/sim_time[count], axis=1)
        mask = df["Arena Size"] == size
        df.loc[mask, "First Passage Time"] = df[mask].apply(lambda row: (row["First Passage Time"]/sim_time[count])*100,axis=1)
        

    return df

def createEBNsFromPostEva(posteva_folder, fpt_folder, num_node, num_robots, evolution_targets_type):
    ebn_parameters_and_fpt = io_scripts.getEBNsParametersFileFromPostEvaFolder(posteva_folder, num_node, num_robots, evolution_targets_type)
    ebn_parameters_and_fpt = io_scripts.readEBNsTsvFptPerformance(fpt_folder, num_node, num_robots, evolution_targets_type, ebn_parameters_and_fpt)

    ebns = []
    for ebn_dict in ebn_parameters_and_fpt:
        ebn = BooleanNetwork.BooleanNetwork(ebn_dict["node"], bn_type="EBN", read_from=ebn_dict["parameters"], net_id=f'{ebn_dict["seed"]:04}')
        ebn.readRobotsInitialStates(num_robots, ebn_dict["initial state"])
        KILOBOT_BIAS = True
        num_trials = 100
        arena_radius = 0.475
        ebn.setPerformanceExperiment(num_robots, num_trials, round((arena_radius-0.0250)*200), KILOBOT_BIAS, fpt=ebn_dict["fpt"])
        ebns.append(ebn)

    return ebns