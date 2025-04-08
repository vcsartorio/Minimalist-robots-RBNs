import time

class LOG(object):
    def __init__(self, log_folder, experiment, num_threads, num_nodes, num_robots, num_networks, num_gen=700):
        date_time = time.strftime("%Y-%m-%d")
        self.log_file_name = f'{log_folder}{date_time}_{experiment}_{num_nodes}N_log_file.txt'
        with open(self.log_file_name, "a+") as log:
            try:
                log.truncate(0)
                if experiment == "GA":
                    log.write("GA - " + str(num_nodes) + " Nodes EBN Simulation ---------------- Start Date: " 
                                + date_time + "\n\n")
                    log.write("Evolution parameters:\n Generations: %d\n Population: %d\n Threads: %d\n" % (num_gen, 
                        num_networks, num_threads))
                    log.write("Simulation parameters:\n Robots: %d\n" % (num_robots))
                else:         
                    log.write("FPT - " + str(num_nodes) + " Nodes RBN Simulation ---------------- Start Date: " + date_time + "\n\n")
                    log.write("Experiment parameters:\n Networks: %d\n Threads: %d\n" % (num_networks, 
                        num_threads))
                    log.write("Simulation parameters:\n Robots: %d\n" % (num_robots))
                log.close()
            except Exception as e:
                print("Couldnt open LOG file! Error: " + str(e))

    def write(self, message, show):
        if show:
            print(message)
        with open(self.log_file_name, "a+") as log:
            try:
                log.write(message)
                log.write("\n")
                log.close()
            except Exception as e:
                print("Couldnt open LOG file!\n" + str(e))