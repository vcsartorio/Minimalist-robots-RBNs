import sys
import scripts.GeneticAlgorithm as GA
import scripts.post_evaluation as post_evaluation
import scripts.network_behaviour_analysis as network_behaviour_analysis
import scripts.fpt_evaluation as fpt_evaluation

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Not enough arguments!")
    else:
        script_name = str(sys.argv[1])
        try:
            if script_name == "GA":
                num_threads = int(sys.argv[2])
                seed_id = int(sys.argv[3])
                num_nodes = int(sys.argv[4])
                print(f"Running {seed_id} seed(s) for Genetic Algorithm.")
                GA.run_evolution(num_threads, seed_id, num_nodes)

            elif script_name == "posteva":
                num_threads = int(sys.argv[2])
                post_evaluation.runPostEvaluation(num_threads)

            elif script_name == "chaos":
                bn_type = sys.argv[2]
                list_nodes = sys.argv[3]
                num_threads = int(sys.argv[4])
                network_behaviour_analysis.main(bn_type, num_threads, list_nodes)

            elif script_name == "fpt_evaluation":
                bn_type = sys.argv[2]
                list_nodes = sys.argv[3]
                num_threads = int(sys.argv[4])
                fpt_evaluation.runFptEvaluation(num_threads, bn_type, list_nodes)

            else:         
                print(f"Script {script_name} not found!")

        except Exception as e:
            print(f"(main.py) Error: {e}")

            
        

