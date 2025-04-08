import random
import math
import os

evoluton_targets = [(0.38,0), (0,0.38), (-0.38,0), (0,-0.38), (0.25,0.0), (0.0,0.25), (-0.25, 0.0), (0.0,-0.25)]

def createTargetPosition(max_trials, read_target_file, arena_radius):
    print("Creating targets position...")
    targets_pos = []
    # if os.path.exists(target_file) and read_target_file:
    #     targets_pos = readTargetPosition()
    # else:
    for t in range(max_trials):
        theta = random.uniform(-math.pi, math.pi)
        rho = random.uniform(0,arena_radius - 0.025)
        pos_xy = (rho * math.cos(theta), rho * math.sin(theta))
        targets_pos.append(pos_xy)

        # saveTargetsPositions(targets_pos, "target.txt")
        # printTargetPositions(targets_pos)

    return targets_pos

def createAllTargetPositions(num_posteva, num_trials, arena_radius):
    # random.seed(10)
    all_target_positions = []
    for count in range(num_posteva):
        all_target_positions.append(createTargetPosition(num_trials, False, arena_radius))
    return all_target_positions

def saveTargetsPositions(targets_pos, target_file):
    with open(target_file, "a+") as target:
        try:
            # target.truncate(0)
            for i in targets_pos:
                target.write("%.3f %.3f\n" % (i[0],i[1]))
                # print("target positions: %.6f %.6f" % (i[0],i[1]))
            target.close()
        except Exception as e:
            print("Couldnt open target file!" + str(e))

    print("Targets position saved!")

def printTargetPositions(target_pos):
    # print("Targets position are: \n")
    print(target_pos)

def readTargetPosition(target_file):
    print("Reading targets position file...")
    target_pos = []
    with open(target_file, "r") as target:
        try:
            lines = target.readlines()
            for line in lines:
                target_pos.append([float(i) for i in line.split()])
                # print("target positions: %.6f %.6f" % (target_pos[-1][0],target_pos[-1][1]))
            target.close()
        except Exception as e:
            print("Couldnt open target file!" + str(e))

    return target_pos

def getEvolutionTarget(trials):
    targets = []
    for i in range(10) :
        targets += evoluton_targets

    return targets