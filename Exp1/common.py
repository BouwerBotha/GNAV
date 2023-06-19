import os




MOVABLE_BOARD_RANGE_X = [-0.72,0.72]
MOVABLE_BOARD_RANGE_Y = [-0.72,0.72]

WIDTH = 64
HEIGHT = 48
NUM_DIM = 3
BETA = 1

NUM_SIMS = 10
POPULATION_SIZE = 100

REAL_BOARD_DIM_X = [-0.92,0.92]
REAL_BOARD_DIM_Y = [-0.92,0.92]

NUM_EVAL_TIMES = 30
ROBOT_STEP_TIME = 0.5
CLOSE_ENOUGH_TO_STOP = 0.12
MAX_TIME_STEPS = 50
SPEED_LIMIT = 0.7

EVAL_MAX_TIME_STEPS = 70

INCL_PREV_COMMANDS = False
COMMAND_STEP_SKIP = 1



def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mkdir_forfile(file):
    mkdir(os.path.dirname(file))