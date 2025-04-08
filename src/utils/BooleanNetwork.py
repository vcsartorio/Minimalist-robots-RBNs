import random
import src.utils.ExperimentPerformance as ExperimentPerformance
from xmlrpc.client import Boolean
import os

bf_min = 0
bf_max = 2

class BooleanNetwork:
    
    def __init__(self, n_nodes, **kwargs):
        self.num_nodes = n_nodes
        self.num_bfunction = n_nodes - 1

        self.bn_type = kwargs.get('bn_type')
        self.encoded = kwargs.get('encoded')
        self.parameters_file = kwargs.get('read_from')
        self.num_connections = kwargs.get('K')
        self.net_id = kwargs.get('net_id')

        if self.parameters_file:
            self.readNetwork()
        elif self.num_connections:
            self.createHomogeneousRBN()
        elif kwargs.get('parameters') is None:
            self.createRandomBooleanNetwork()
        else:
            self.setParameters(kwargs.get('parameters'))

    def createRandomBooleanNetwork(self):
        self.connections = []
        self.bfunctions = []
        for i in range(self.num_nodes):
            self.connections.append([])
            for j in range(self.num_nodes):
                self.connections[i].append(random.randint(bf_min, bf_max))
        for i in range(self.num_nodes):
            self.bfunctions .append([])
            for j in range(self.num_bfunction):
                self.bfunctions [i].append(random.randint(bf_min, bf_max))

    def createHomogeneousRBN(self):
        self.connections = []
        self.bfunctions = []
        for i in range(self.num_nodes):
            self.connections.append([])
            for j in range(self.num_nodes):
                self.connections[i].append(0)
            k_homogeneous = random.sample(range(0, self.num_nodes), self.num_connections)
            for j in k_homogeneous:
                self.connections[i][j] = random.randint(1,2)
        for i in range(self.num_nodes):
            self.bfunctions.append([])
            for j in range(self.num_bfunction):
                self.bfunctions[i].append(random.randint(bf_min, bf_max))

    def readNetwork(self):
        self.connections = []
        self.bfunctions = []
        with open(self.parameters_file, "r") as parameters:
            try:
                for i in range(self.num_nodes):
                    self.connections.append([])
                    line = parameters.readline()
                    for ch in line:
                        if ch != '\n':
                            self.connections[i].append(int(ch))
                for i in range(self.num_nodes):
                    self.bfunctions.append([])
                    line = parameters.readline()
                    for ch in line:
                        if ch != '\n':
                            self.bfunctions[i].append(int(ch))
                parameters.close()
            except Exception as e:
                print("Error opening network parameters file. " + str(e))
                exit(1)

    def setParameters(self, parameters):
        self.connections = []
        for i in range(self.num_nodes):
            self.connections.append([])
            for j in range(self.num_nodes):
                self.connections[-1].append(parameters[i+j])
        start_idx = self.num_nodes*self.num_nodes
        self.bfunctions = []
        for i in range(self.num_nodes):
            self.bfunctions.append([])
            for j in range(self.num_nodes-1):
                self.bfunctions[-1].append(parameters[start_idx+i+j])

    def resetNetworkParameters(self, connections, bfunctions):
        self.connections = connections
        self.bfunctions = bfunctions

    def printNetworkParameters(self):
        print("Connections:")
        for node_connections in self.connections:
            node_connections = [str(c) for c in node_connections]
            print("".join(node_connections), end="")
        print("\nBolean Functions:")
        for node_bfunctions in self.bfunctions:
            node_bfunctions = [str(bf) for bf in node_bfunctions]
            print("".join(node_bfunctions), end="")
        print("\n")

    ### ---------------------------------------------------------- Boolean Network Functions ------------------------------------------------------- ###

    def setDeltaValue(self, delta):
        self.average_delta = delta

    def setLMCValues(self, comp, ent, diseq):
        self.average_complexity = comp
        self.average_entropy = ent
        self.average_disequilibrium = diseq

    def setStepLength(self, avg_sl):
        self.total_average_sim_sl = avg_sl

    def generateInitialState(self):
        self.initial_state = []
        for i in range(self.num_nodes):
            self.initial_state.append(random.randint(0,1))

        return self.initial_state.copy()

    def flipNodeinInitialState(self, num_flip_nodes):
        for f in range(num_flip_nodes):
            node = random.randint(0,self.num_nodes-1)
            self.initial_state[node] ^= 1

        return self.initial_state.copy()

    def getNetworkStatesOverTimeSteps(self, time_step):
        self.node_states = [list(self.initial_state)]
        self.node_decimal_values = []
        self.angle_values = []
        self.step_length_values = []

        self.connections_number = self.contConnectionsNum()
        self.calculateDecimalValue(self.node_states[0])
        self.calculateSimulationValues(self.node_states[0])

        for t in range(time_step):
            new_state = self.calculateNextStep()
            self.node_states.append(new_state)
            self.calculateDecimalValue(new_state)
            self.calculateSimulationValues(new_state)

        self.average_step_length = round(sum(self.step_length_values)/len(self.step_length_values))

        return self.node_states, self.node_decimal_values

    def getNetworkStatesOverSimTime(self, sim_time):
        self.node_states = [list(self.initial_state)]
        self.node_decimal_values = []
        self.angle_values = []
        self.step_length_values = []

        self.connections_number = self.contConnectionsNum()
        self.calculateDecimalValue(self.node_states[0])
        self.calculateSimulationValues(self.node_states[0])

        max_turning_ticks = 160
        elapse_time = 0
        while elapse_time < sim_time:
            new_state = self.calculateNextStep()
            self.node_states.append(new_state)
            self.calculateDecimalValue(new_state)
            self.calculateSimulationValues(new_state)
            elapse_time += (abs(self.angle_values[-1])*max_turning_ticks + self.step_length_values[-1])

        self.average_sim_step_length = round(sum(self.step_length_values)/len(self.step_length_values))
        # print("Total time steps: %d" % (len(self.step_length_values)))

        return self.node_states, self.node_decimal_values

    def getAvgSLUntilSimulationEnd(self, time):
        sl_from_sim = []
        elapse_time_sim = 0
        max_turning_ticks = 160
        for t in range(len(self.step_length_values)):
            sl_from_sim.append(self.step_length_values[t])
            elapse_time_sim += (abs(self.angle_values[-1])*max_turning_ticks + self.step_length_values[t])
            if elapse_time_sim > time:
                break

        self.average_sim_step_length = round(sum(sl_from_sim)/len(sl_from_sim))
        return self.average_sim_step_length

    def calculateDecimalValue(self, state):
        value = 0
        for i in range(self.num_nodes):
            if state[i]:
                value += pow(2, i)
        self.node_decimal_values.append(value)

    def calculateSimulationValues(self, new_state):
        angle_direction = 0
        num_values = int(self.num_nodes/2)
        for i in range(num_values):
            if new_state[i] == 1:
                angle_direction += pow(2, i)

        angle = angle_direction - ((pow(2, num_values-1) - 1))
        angle = (angle / (float(pow(2, num_values-1) - 1)))

        step_lenght = 0
        for i in range(num_values, self.num_nodes):
            if new_state[i] == 1:
                step_lenght += pow(2, i - num_values)

        step_lenght = step_lenght/32
        
        self.angle_values.append(angle)
        self.step_length_values.append(step_lenght)

    def contConnectionsNum(self):
        connections_number = []
        for connected_list in self.connections:
            cont_connections = 0
            for connected_node in connected_list:
                if connected_node:
                    cont_connections += 1
            connections_number.append(cont_connections)
        return connections_number
    
    def getNodeState(self):
        return self.node_states[-1]

    def getNodeValue(self):
        return self.node_decimal_values[-1]

    def logicMap(self, function, node_a, node_b):
        if function == 0:
            return (node_a & node_b) # AND    
        elif function == 1:
            return (node_a | node_b) # OR
        elif function == 2:
            return (node_a ^ node_b) # XOR
        elif function == 3:
            return ~(node_a ^ node_b) # XNOR
        elif function == 4:
            return ~(node_a & node_b) # NAND
        elif function == 5:
            return ~(node_a | node_b) # NOR
        else:
            print("Error in logicmap = %d\n" % function)
        return 0

    def calculateNextStep(self):
        new_node_states = []

        # Calculate the next step for each node
        for i in range(self.num_nodes):

            # Get the state of each node connected to it
            if self.connections_number[i] > 0:
                connected_nodes = []
                for j in range(self.num_nodes):
                    if self.connections[i][j] == 1:
                        connected_nodes.append(self.getNodeState()[j])
                    elif self.connections[i][j] == 2:
                        if self.getNodeState()[j]:
                            connected_nodes.append(0)
                        else:
                            connected_nodes.append(1)

                # Calculate next node state
                flogic_num = self.connections_number[i] - 1
                cont_states = 0
                for j in range(flogic_num):
                    aux_flogic = self.bfunctions[i][j]
                    new_state = self.logicMap(aux_flogic, connected_nodes[cont_states], connected_nodes[cont_states+1])
                    connected_nodes.append(new_state)
                    cont_states += 2
                new_node_states.append(connected_nodes[-1])
            else:
                new_node_states.append(self.getNodeState()[i])

        return new_node_states

    ### --------------------------------------------------------- Experiment Functions ----------------------------------------------------- ###

    def setPerformanceExperiment(self, num_robots, num_trials, arena_radius, bias, **kwargs):
        self.fpt_result = kwargs.get('fpt')
        self.experiment_performance = ExperimentPerformance.ExperimentPerformance(num_robots, num_trials)
        if kwargs.get('save_exp'):
            self.experiment_date = kwargs.get('date')
            self.performance_file = "%s_%dR_%dN_%dcm%s%s%s.tsv" % (self.bn_type, num_robots, self.num_nodes, arena_radius, 
                "_NoBias" if not bias else "" , "_encoded" if self.encoded == "encoded" else "", self.experiment_date if self.experiment_date else "")
            self.experiment_performance.initializeExperiment(self.performance_file, self.net_id)

    def resetExperimentResults(self):
        self.experiment_performance.resetResults()

    def createExperimentParametersFile(self):
        with open("src/behaviors/parameters_rbn.txt", "w+") as parameters:
            try:
                parameters.truncate(0)
                parameters.write("%d\n" % self.num_nodes)
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        parameters.write("%d" % self.connections[i][j])
                    parameters.write("\n")
                for i in range(self.num_nodes):
                    for j in range(self.num_bfunction):
                        parameters.write("%d" % self.bfunctions[i][j])
                    parameters.write("\n")
                for i in range(len(self.robots_initial_states)):
                    for j in range(self.num_nodes):
                        parameters.write("%d" % self.robots_initial_states[i][j])
                    parameters.write("\n")
                parameters.close()
            except Exception as e:
                print("Couldnt open parameters file!\n" + str(e))

    def saveNetworkParametersAndPerformance(self, parameters_folder):
        network_file = "%s_%dR_%dN%s_%s_parameters_and_fpt.txt" % (self.bn_type, len(self.robots_initial_states), self.num_nodes, 
            "_encoded" if self.encoded == "encoded" else "", self.net_id)
        file_path = parameters_folder + network_file
        if os.path.exists(file_path):
            print("Warning! Parameters Files already created!")
        else:
            with open(file_path, "a+") as parameters:
                try:
                    fpt_weibull_results = [str(x) for x in self.experiment_performance.weibull_disc_evaluations_value]
                    fpt_weibull_results = ' '.join(fpt_weibull_results)
                    parameters.write("fpt: %s\n" % (fpt_weibull_results))
                    for i in range(self.num_nodes):
                        for j in range(self.num_nodes):
                            parameters.write("%d" % self.connections[i][j])
                        parameters.write("\n")
                    for i in range(self.num_nodes):
                        for j in range(self.num_bfunction):
                            parameters.write("%d" % self.bfunctions[i][j])
                        parameters.write("\n")
                    for i in range(len(self.robots_initial_states)):
                        for j in range(self.num_nodes):
                            parameters.write("%d" % self.robots_initial_states[i][j])
                        parameters.write("\n")
                    parameters.write("\n")
                    parameters.close()
                except Exception as e:
                    print("Couldnt save network parameters file!\n" + str(e))
    
    def createRobotsInitialStates(self, num_robots):
        self.robots_initial_states = []
        for i in range(num_robots):
            self.robots_initial_states.append(self.generateInitialState())

        return self.robots_initial_states

    def setRobotsInitialState(self, initial_states):
        self.robots_initial_states = []
        for i in initial_states:
            self.robots_initial_states.append(i)

    def readRobotsInitialStates(self, num_states, initial_states_file):
        self.robots_initial_states = []
        with open(initial_states_file, "r+") as init_states:
            try:
                for i in range(num_states):
                    self.robots_initial_states.append([])
                    line = init_states.readline()
                    for ch in line:
                        if ch != '\n':
                            self.robots_initial_states[i].append(int(ch))
                init_states.close()
            except Exception as e:
                print("Error opening network initial state file. " + str(e))
                exit(1)

        return self.robots_initial_states

    def saveInitialState(self):
        initial_state_file = "%s_%sinitial_state_%dR_%dN.txt" % (self.bn_type, self.encoded, len(self.robots_initial_states), self.num_nodes)
        with open(initial_state_file, "w+") as init_state:
            try:
                init_state.truncate(0)
                for i in range(len(self.robots_initial_states)):
                    for j in range(len(self.robots_initial_states[i])):
                        init_state.write("%d" % (self.robots_initial_states[i][j]))
                    init_state.write("\n")
                init_state.close()
            except Exception as e:
                print("Couldnt write initial states!\n" + str(e))
