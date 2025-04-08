import src.utils.BooleanNetwork as BooleanNetwork
import src.ArgosSimulation as ArgosSimulation
import os
import time

class NetworkKilobotsExperiment(object):

    parameters_folder = "Data/"

    class KilobotSimulation(object):

        def __init__(self, sim_id, trial, process):
            self.sim_id = sim_id
            self.trial = trial
            self.process = process
            self.start_time = time.time()
            self.simulation_total_time = -1

        def simulationHasEnd(self):
            status = self.process.poll()
            if status is not None:
                self.simulation_total_time = time.time() - self.start_time
                return True
            
            return False

        def checkSimulationTime(self):
            status = self.process.poll()
            if status is not None:
                self.simulation_total_time = time.time() - self.start_time
                if self.simulation_total_time > 120:
                    print("Simulation taking too long to finished")

        def printSimulationTotalTime(self):
            print(f'Total time: {self.simulation_total_time}s')

        def __repr__(self):
            return f'{self.sim_id:04}#{self.trial:03}'

    def __init__(self, num_threads, num_robots, targets_position, arena_radius, simulation_time, bias, save_pos, save_parameters):
        self.num_threads = num_threads
        self.num_robots = num_robots
        self.targets_position = targets_position
        self.arena_radius = arena_radius
        self.simulation_time = simulation_time
        self.save_pos = save_pos
        self.save_parameters = save_parameters
        self.kilobot_bias = bias

    def changeTargetPositions(self, new_targets):
        self.targets_position = new_targets

    def executeKilobotExperimentTrials(self, networks):
        simulation_pool = []
        last_parameters_file = ""
        while True:
            last_parameters_file = self.checkNetworkTrials(networks, simulation_pool, last_parameters_file)
            self.checkSimulationPool(simulation_pool, networks)
            experiment_end = self.checkNetworkFinalFitness(networks)
            if experiment_end:
                break

        for network in networks:
            network.experiment_performance.printResult()
            if self.save_parameters:
                network.saveNetworkParametersAndPerformance(self.parameters_folder)

    def checkNetworkTrials(self, networks, simulation_pool, last_parameters_file):
        for network in networks:
            if network.experiment_performance.num_trials < network.experiment_performance.max_trials:
                last_parameters_file = self.addSimulationOnPool(simulation_pool, network, last_parameters_file)
                break

        return last_parameters_file  

    def addSimulationOnPool(self, simulation_pool, network, last_parameters_file):
        if len(simulation_pool) < self.num_threads and not os.path.exists(last_parameters_file):
            trial = network.experiment_performance.num_trials
            sim_id = network.net_id + f'{trial:03}'
            network.createExperimentParametersFile()
            last_parameters_file, process = ArgosSimulation.callArgosSimulation(network.bn_type, self.num_robots, network.num_nodes, 
                self.targets_position[trial], sim_id, self.arena_radius, self.simulation_time, self.save_pos, self.kilobot_bias)
            simulation_process = self.KilobotSimulation(int(network.net_id), trial, process)
            simulation_pool.append(simulation_process)
            print(f'Running {int(network.net_id)+1} Network -> {trial+1} trial! Active Threads: {len(simulation_pool)}')
            # print(f'Running {int(network.net_id)+1} Network -> {trial+1} trial! Threads: {simulation_pool}')
            network.experiment_performance.num_trials += 1

        return last_parameters_file

    def checkSimulationPool(self, simulation_pool, networks):
        process_has_end = False
        process_idx = -1
        for idx, simulation in enumerate(simulation_pool):
            process_has_end, sim_results = ArgosSimulation.checkProcessStatus(simulation, self.num_robots)
            if process_has_end:
                for network in networks:
                    if int(network.net_id) == simulation.sim_id:
                        network.experiment_performance.setFitnessValues(sim_results['disc'], sim_results['inf'], sim_results['frac disc'], 
                            sim_results['frac inf'], sim_results['disc robots'], simulation.trial)
                        process_idx = idx
                        # print("%d Network %d trial is finished! %s scores: %d Discovery Time with %d%% Fraction discovery" % (simulation.sim_id+1, simulation.trial+1, 
                        #     network.bn_type, sim_results['disc'], sim_results['frac disc']*100))
                        # simulation.printSimulationTotalTime()
                        break
                if process_idx != -1:
                    break
                else:
                    print("Error! Simulation id dont belong to any network!")

        if process_has_end:
            del simulation_pool[process_idx]

    def checkNetworkFinalFitness(self, networks):
        experiment_end = True
        for network in networks:
            if not network.experiment_performance.computed_final_fitness:
                experiment_end = False

        return experiment_end