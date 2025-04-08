import os
import time
import random
import csv
# from pymoo.configuration import Configuration
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
# import pymoo.operators.sampling.rnd
import numpy as np
import autograd.numpy as anp
import scripts.log_script as log_script
import src.utils.BooleanNetwork as BooleanNetwork
import src.utils.Targets as Targets
import src.KilobotsSearchExperiment as KilobotsSearchExperiment

#### CONFIGURATION PARAMETERS #####
fixed_targets_positions = [(0.38,0), (0,0.38), (-0.38,0), (0,-0.38), (0,0.25), (0.25,0), (0,-0.25), (-0.25,0)]

# Path to results files
best_individuals_data = "best_individuals_data.txt"
evolution_curve_data = "evolution_curve_data.tsv"
initial_state_file = "initial_state_file.txt"
population_file = "population.txt"

class Experiment:

    def __init__(self, max_g, n_pop, n_threads, n_trials, n_nodes, n_robots, target_type, targets, arena_radius):
        # GA parameters
        self.max_generation = max_g
        self.num_population = n_pop
        self.max_trials = n_trials
        self.num_threads = n_threads
        self.min_fracdisc = 0.2
        
        # Argos Simulation parameters
        self.num_robots = n_robots
        self.arena_radius = arena_radius
        self.targets = targets
        self.target_type = target_type
        self.simulation_time = 3000
        self.bias = True
        
        # Random Boolean Network parameters
        self.num_nodes = n_nodes
        self.num_boolfunction = self.num_nodes - 1
        self.cn_maxvalue = 2
        self.cn_minvalue = 0
        self.bf_maxvalue = 2
        self.bf_minvalue = 0
        self.num_variables = (self.num_nodes*self.num_nodes) + (self.num_nodes*(self.num_nodes-1))

    def nameDataDir(self, seed_id):
        self.data_dir = f"/input/Evolutions/{time.strftime('%Y-%m-%d')}_ga_{self.num_robots}R_{self.num_nodes}N_{self.target_type}-targets_{seed_id}seed/"

    def setStartTime(self, time):
        self.start_time = time
    
class MyProblem(Problem):

    def __init__(self, experiment):
        self.experiment = experiment

        cn_xl = self.experiment.cn_minvalue * anp.ones(self.experiment.num_nodes*self.experiment.num_nodes)
        bf_xl = self.experiment.bf_minvalue * anp.ones(self.experiment.num_nodes*(self.experiment.num_nodes-1))
        xl = anp.append(cn_xl, bf_xl)
        cn_xu = self.experiment.cn_maxvalue * anp.ones(self.experiment.num_nodes*self.experiment.num_nodes)
        bf_xu = self.experiment.bf_maxvalue * anp.ones(self.experiment.num_nodes*(self.experiment.num_nodes-1))
        xu = anp.append(cn_xu, bf_xu)

        self.initial_states = self.createInitialStates(save_states=True)
        # self.initial_states = self.readInitialStates()

        self.generation = 0
        self.best_fitness = 1000000
        self.target_positions = []
        # random.seed()

        self.run_experiment = KilobotsSearchExperiment.NetworkKilobotsExperiment(self.experiment.num_threads, self.experiment.num_robots, 
            self.target_positions, self.experiment.arena_radius, self.experiment.simulation_time, self.experiment.bias, False, False)
        super().__init__(self.experiment.num_variables, n_obj=1, n_ieq_constr=1, xl=xl, xu=xu, evaluation_of="auto")

    def _evaluate(self, population, out, *args, **kwargs):
        self.LOG.write("\nStarting Generation %d..." % (self.generation + 1), True)
        rbns = self.setPopulationNetwork(population)
        # self.printPopulationParameters(rbns)
        self.checkExperimentTargets()

        self.LOG.write("Starting Experiment with population network...", True)
        self.run_experiment.executeKilobotExperimentTrials(rbns)

        out["F"] = self.getPopulationFitness(rbns)
        out["G"] = self.experiment.min_fracdisc - self.getPopulationFractionDiscovery(rbns)
        # self.printPopulationParameters(rbns)
        self.savingGeneration(rbns)
        self.savingIndividualEvolutionCurve(rbns)
        self.generation += 1
        self.LOG.write(f"Evaluation step is finished! Total Time: {round(time.time() - self.experiment.start_time, 2)} seconds", True)
        
    def setPopulationNetwork(self, population_parameters):
        self.LOG.write("Setting network population...", True)
        rbns = []
        for count_networks in range(self.experiment.num_population):
            # print(population_parameters[count_networks])
            rbn = BooleanNetwork.BooleanNetwork(self.experiment.num_nodes, bn_type="EBN", parameters=population_parameters[count_networks], 
                net_id=f'{count_networks:04}')
            rbn.setRobotsInitialState(self.initial_states)
            rbn.setPerformanceExperiment(self.experiment.num_robots, self.experiment.max_trials, self.experiment.arena_radius, self.experiment.bias, save_exp=False)
            rbns.append(rbn)
            # rbn.printNetworkParameters()

        return rbns
    
    def getPopulationFitness(self, rbns):
        self.LOG.write("Generation %d finished. Getting population fitness..." % (self.generation + 1), True)
        pop_fitness = anp.zeros(self.experiment.num_population)
        for idx in range(len(rbns)):
            pop_fitness[idx] = rbns[idx].experiment_performance.weibull_discovery_time

        return pop_fitness

    def getPopulationFractionDiscovery(self, rbns):
        self.LOG.write("Checking population Fraction Discovery...", True)
        pop_fracdisc = anp.zeros(self.experiment.num_population)
        for idx in range(len(rbns)):
            # pop_fracdisc[idx] = random.random()
            pop_fracdisc[idx] = rbns[idx].experiment_performance.fraction_discovery
            
        return pop_fracdisc
    
    def checkExperimentTargets(self):
        self.LOG.write("Checking simulation targets...", True)
        if (self.experiment.target_type == 'fixed'):
            self.target_positions = self.experiment.targets            
        elif(self.experiment.target_type == 'fixed_and_random'):
            self.target_positions = self.experiment.targets + Targets.createTargetPosition(self.experiment.max_trials - len(self.experiment.targets), 
                False, self.experiment.arena_radius)
        else:
            self.target_positions = Targets.createTargetPosition(self.experiment.max_trials, False, self.experiment.arena_radius)
            
        self.run_experiment.changeTargetPositions(self.target_positions)
        # self.LOG.write(f"Targets: {self.target_positions}", True)

    def createInitialStates(self, save_states=False):
        initial_states = []
        # random.seed(10)
        for j in range(self.experiment.num_robots):
            initial_states.append([])
            for k in range(self.experiment.num_nodes):
                initial_states[j].append(random.randint(0,1))

        if save_states:
            with open(self.experiment.data_dir + initial_state_file, "a+") as init_file:
                try:
                    init_file.truncate(0)
                    for i in initial_states:
                        for j in i:
                            init_file.write(str(j))
                        init_file.write("\n")
                    init_file.close()
                except Exception as e:
                    self.LOG.write("Error! Couldnt save initial state:" + str(e), True)

        return initial_states

    def readInitialStates(self):
        initial_states = []
        with open(self.experiment.data_dir + initial_state_file, "r+") as init_states:
            try:
                for i in range(self.experiment.num_robots):
                    initial_states.append([])
                    line = init_states.readline()
                    for ch in line:
                        if ch != '\n':
                            initial_states[i].append(int(ch))
                init_states.close()
            except Exception as e:
                self.LOG.write("Couldnt open initial state file!" + str(e), True)
                exit(1)
        
        return initial_states

    def savingIndividualEvolutionCurve(self, population):
        best_gen_weibull = 100000
        best_idx = 0
        for i in range(len(population)):
            if best_gen_weibull > population[i].experiment_performance.weibull_discovery_time:
                best_gen_weibull = population[i].experiment_performance.weibull_discovery_time
                best_idx = i

        file_path = self.experiment.data_dir + evolution_curve_data
        if not os.path.exists(file_path):
            with open(file_path, "wt") as out_file:
                try:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    tsv_writer.writerow(['Id', 'Weibull Discovery Time', 'Discovery Time', 'Fraction Discovery', 'Information Time', 'Fraction Information'])
                    tsv_writer.writerow([best_idx, round(population[best_idx].experiment_performance.weibull_discovery_time, 0), 
                        round(population[best_idx].experiment_performance.discovery_time, 0), round(population[best_idx].experiment_performance.fraction_discovery, 4), 
                        round(population[best_idx].experiment_performance.information_time, 0), round(population[best_idx].experiment_performance.fraction_information, 4)])
                    out_file.close()
                except Exception as e:
                    self.LOG.write("Error! Couldnt open evolution curve file for saving %d as best individual from %d gen" % (best_gen_weibull, self.generation+1), True)
                    self.LOG.write(str(e), True)
        else:
            with open(file_path, "a+") as out_file:
                try:
                    tsv_writer = csv.writer(out_file, delimiter='\t')
                    tsv_writer.writerow([best_idx, round(population[best_idx].experiment_performance.weibull_discovery_time, 0), 
                        round(population[best_idx].experiment_performance.discovery_time, 0), round(population[best_idx].experiment_performance.fraction_discovery, 4), 
                        round(population[best_idx].experiment_performance.information_time, 0), round(population[best_idx].experiment_performance.fraction_information, 4)])
                    out_file.close()
                except Exception as e:
                    self.LOG.write("Error! Couldnt open evolution curve file for saving %d as best individual from %d gen" % (best_gen_weibull, self.generation+1), True)
                    self.LOG.write(str(e), True)

        self.savingIndividual(population[best_idx], best_idx)

    def savingIndividual(self, individual, individual_idx):
        with open(self.experiment.data_dir + best_individuals_data, "a+") as idata:
            try:
                idata.write("fitness: %d %d %f %d %f Gen: %d\n" % (individual.experiment_performance.weibull_discovery_time, individual.experiment_performance.discovery_time, 
                    individual.experiment_performance.fraction_discovery, individual.experiment_performance.information_time, individual.experiment_performance.fraction_information, self.generation+1))
                for connections in individual.connections:
                    str_connections = [str(c) for c in connections]
                    str_connections = "".join(str_connections)
                    idata.write(f'{str_connections}\n')
                for bfunctions in individual.bfunctions:
                    str_bfunctions = [str(bf) for bf in bfunctions]
                    str_bfunctions = "".join(str_bfunctions)
                    idata.write(f'{str_bfunctions}\n')
                idata.write("\n")
                idata.close()
            except Exception as e:
                self.LOG.write("Error! Couldnt save %d as best individual from %d gen in '%s' file" % (individual_idx, self.generation+1, best_individuals_data), True)
                self.LOG.write(str(e), True)

    def savingGeneration(self, population):
        file_name = self.experiment.data_dir + population_file
        with open(file_name, "a+") as idata:
            try:
                idata.truncate(0)
                idata.write("Generation %d:\n" % (self.generation))
                for individual in population:
                    for connections in individual.connections:
                        str_connections = [str(c) for c in connections]
                        str_connections = "".join(str_connections)
                        idata.write(f'{str_connections}\n')
                    for bfunctions in individual.bfunctions:
                        str_bfunctions = [str(bf) for bf in bfunctions]
                        str_bfunctions = "".join(str_bfunctions)
                        idata.write(f'{str_bfunctions}\n')
                    idata.write("\n")
                idata.close()
            except Exception as e:
                print("Error saving generation! Output: " + str(e))

    def printPopulationParameters(self, rbns):
        print("Generation %d:" % (self.generation + 1))
        for i in range(self.experiment.num_population):
            self.printIndividualParameters(rbns[i], i)

    def printIndividualParameters(self, individual, idx):
        print("Individual %d:" % (idx))
        individual.printNetworkParameters()

    def printPopulationFitness(self, rbns):
        print("Generation %d:" % (self.generation + 1))
        for i in range(self.experiment.num_population):
            self.printIndividualFitness(rbns[i], i)

    def printIndividualFitness(self, individual, idx):
        print("%d Individual fitness is: %d discovery time, %f fraction discovery, %d information time and %F fraction information - %d Weibull discovery time" % (idx, individual.experiment_performance.discovery_time,
            individual.experiment_performance.fraction_discovery, individual.experiment_performance.information_time, individual.experiment_performance.fraction_information, individual.experiment_performance.weibull_discovery_time))

    def enableLOG(self, log):
        self.LOG = log

### ------------------------------------------------------------- Main Functions ---------------------------------------------------------------###
def createDataDir(experiment):
    if not os.path.exists(experiment.data_dir):
        os.makedirs(experiment.data_dir)

def readLastPopulation(folder):
    pop = []
    with open(folder + population_file, "r") as pop_file:
            try:
                lines = pop_file.readlines()
                pop.append([])
                for line in lines[1:]:
                    if line == "\n":
                        if len(pop) > 0:
                            pop[-1] = list(map(int, pop[-1]))
                        pop.append([])
                    else:
                        pop[-1] += line.replace("\n","")

                pop[-1] = list(map(int, pop[-1]))
                pop_file.close()
            except Exception as e:
                print("Couldnt open population file:" + str(e))

    return pop[:-1]

def samplingNewPopulation(experiment_parameters):
    population = []
    for individual in range(experiment_parameters.num_population):
        population.append([])
        for i in range(experiment_parameters.num_nodes):
            for j in range(experiment_parameters.num_nodes):
                population[-1].append(random.randint(experiment_parameters.cn_minvalue, experiment_parameters.cn_maxvalue))
        for i in range(experiment_parameters.num_nodes):
            for j in range(experiment_parameters.num_boolfunction):
                population[-1].append(random.randint(experiment_parameters.bf_minvalue, experiment_parameters.bf_maxvalue))

    return population

def my_callback(algorithm):
    pop = algorithm.res.F
    print(pop)

def run_evolution(num_threads, seed_id, n_nodes):
    start_time = time.time()

    num_nodes = n_nodes
    max_gen = 700
    num_pop = 40
    num_trials = 8
    num_robots = 20
    target_type = "random"
    arena_radius = 0.475
    targets = fixed_targets_positions
    start_from_last_population = False

    # random.seed()
    experiment_parameters = Experiment(max_gen, num_pop, num_threads, num_trials, num_nodes, num_robots, target_type, targets, arena_radius)
    experiment_parameters.nameDataDir(seed_id)
    experiment_parameters.setStartTime(start_time)

    # Configuration.show_compile_hint = False
    createDataDir(experiment_parameters)
    log = log_script.LOG(experiment_parameters.data_dir, "GA", num_threads, num_nodes, num_robots, num_pop, num_gen=max_gen)

    problem = MyProblem(experiment_parameters)
    problem.enableLOG(log)
    
    if not start_from_last_population:
        pop = Population.new("X", samplingNewPopulation(experiment_parameters))
        algorithm = GA(pop_size=experiment_parameters.num_population,
                seed=None,
                sampling=pop,
                crossover=SBX(prob=0.5, eta=10, repair=RoundingRepair()),
                mutation=PolynomialMutation(prob=0.05, eta=25, repair=RoundingRepair()),
                eliminate_duplicates=True)
    else:
        pop = Population.new("X", readLastPopulation(experiment_parameters.data_dir))
        Evaluator().eval(problem, pop)
        algorithm = GA(pop_size=experiment_parameters.num_population,
                    seed=None,
                    sampling=pop,
                    crossover=SBX(prob=0.5, eta=10, repair=RoundingRepair()),
                    mutation=PolynomialMutation(prob=0.05,eta=25, repair=RoundingRepair()),
                    eliminate_duplicates=True)

    res = minimize(problem, 
                algorithm, 
                ('n_gen', experiment_parameters.max_generation),
                seed = None,
                erbose=False)

    print("\nEvolution is finished after %d generations!\n" % experiment_parameters.max_generation)
    # Scatter().add(res.F).show()

    print("Time running: %s seconds" % (round(time.time() - start_time, 2)))