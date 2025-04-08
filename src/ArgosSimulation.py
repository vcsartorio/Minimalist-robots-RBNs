import time
import re
from subprocess import Popen, PIPE
import os
from lxml import etree

# Path to simulation files
parameters_file = "src/behaviors/parameters_rbn.txt"
argos_folder_path = "simulation_config/"
argos_base_file = "simulation_parameters.argos"
argos_base_file_bias = "simulation_parameters_bias.argos"
# argos_base_file_bias = "simulation_parameters_new_encode.argos"
argos_simulation_file = "simulation_parameters.argos"

def callArgosSimulation(strategy, num_robots, num_nodes, target_pos, sim_id, arena_radius, total_time, save_pos, bias):
    # cwd = ""
    # start_path = os.path.dirname(os.getcwd())
    parser = etree.XMLParser(remove_comments = False)
    etree.set_default_parser(parser)
    if bias:
        tree = etree.parse(argos_folder_path + argos_base_file_bias)
    else:
        tree = etree.parse(argos_folder_path + argos_base_file)
    root = tree.getroot()
    for framework in root.iter('framework'):
        framework.find('experiment').set("length", "%d" % total_time)
    for loop_functions in root.iter('loop_functions'):
        loop_functions.set("strategy", "%s" % strategy.upper())
        loop_functions.set("num_nodes", "%d" % num_nodes)
        loop_functions.set("num_robots", "%d" % num_robots)
        # loop_functions.set("num_robots", "%d" % 1)
        loop_functions.set("target_posx", "%.6f" % target_pos[0])
        loop_functions.set("target_posy", "%.6f" % target_pos[1])
        loop_functions.set("id_simulation", "%s" % sim_id)
        loop_functions.set("arena_radius", "%.3f" % arena_radius)
        loop_functions.set("save_position", "%d" % (1 if save_pos else 0))
    id_simu_path = argos_folder_path + str(sim_id) + argos_simulation_file
    tree.write(id_simu_path, xml_declaration = True)
    while True:
        try:
            process = Popen(["argos3", "-c", id_simu_path], stdout=PIPE, stderr=PIPE)
            # argos_command = "argos3 -c " + id_simu_path
            # subprocess.Popen(["argos3", "-c", id_simu_path], stdout=subprocess.PIPE, cwd="../")
            # subprocess.Popen(["xdg-open", id_simu_path])
            # os.system('gnome-terminal -- ' + argos_command)
            break
        except Exception as e:
            print("Couldnt call argos simulation!\n" + str(e))
            time.sleep(1)

    return id_simu_path, process

def checkProcessStatus(simulation, num_robots):
    process_has_end = False
    sim_results = dict()
    if simulation.simulationHasEnd():
        try:
            output, error = simulation.process.communicate()
            if simulation.process.returncode != 0: 
                raise Exception("Simulation %d failed! Process Error: %s" % (simulation.process.pid, error))
            else:
                out_decoded = output.decode('utf-8')
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                results = ansi_escape.sub('', out_decoded).split('\n')
                lines = results[-1].split(' ')[:-1]
                if len(lines) == (4 + num_robots):
                    sim_results['disc'] = int(lines[0])
                    sim_results['inf'] = int(lines[1])
                    sim_results['frac disc'] = float(lines[2])
                    sim_results['frac inf'] = float(lines[3])
                    sim_results['disc robots'] = []
                    for line in lines[4:]:
                        sim_results['disc robots'].append(int(line))
                    process_has_end = True
                else:
                    raise Exception("Fitness file of %d individual %d trial has %d lines" % (simulation.sim_id, simulation.trial, len(lines)))
        except Exception as e:
            print("Code Exception: %s" % (str(e)))
            exit(1)

    return process_has_end, sim_results
