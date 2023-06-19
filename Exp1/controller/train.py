import random
import time
import math
import tensorflow as tf
import os
import common
import controller.network
from common import NUM_SIMS,MOVABLE_BOARD_RANGE_X,MOVABLE_BOARD_RANGE_Y,CLOSE_ENOUGH_TO_STOP,MAX_TIME_STEPS,BETA,HEIGHT,WIDTH,NUM_DIM,POPULATION_SIZE
import numpy as np
import Simulator.request
import SNN.network
import network
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
from GAN.simple_network import get_generator,get_discriminator,CycleGan,generator_loss_fn,discriminator_loss_fn

gen_G = get_generator(name="generator_G")
gen_F = get_generator(name="generator_G")


disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate = 0.0002,beta_1 = 0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate = 0.0002,beta_1 = 0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate = 0.0002,beta_1 = 0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate = 0.0002,beta_1 = 0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

cycle_gan_model.load_weights('../GAN/model_checkpoints/3/cyclegan_checkpoints.100')
generator = cycle_gan_model.gen_G
cycle_gan_model = None
NUM_SCENES = 10

SNN_NOISE_SCALE = 0.075
MUTATION_SIZE = 0.03#was 0.01
MUTATION_PROB = 0.5
NUM_EVOLUTIONS = 2000
TOURNAMENT_SIZE = 5
ELITISM = True
MULTIPLICATIVE_BAD_PENALTY= 10.0
SAVE_EVERY = 20
CHECKPOINT_EVERY = 50
SAVE_CONTROLLER_EVERY = 200
META_DATA_PATH = '../data/meta'
SNN_PATH = '../SNN/checkpoints/snn.ckpt'
SAVE_STRING = 'saved_controllers/objects_{0}_{1}_{2}_atthead_IMPORTANCE_Patch_SIZE_{3}'.format('GRU' if network.GRU else 'LSTM',network.cells[0],network.NUM_HEADS,network.PATCH_SIZE)
EXTRACT_DATA_EVERY = 5
HEADLESS = 50

motion_sim = SNN.network.get_model()
motion_sim.load_weights(SNN_PATH)



PLOT_STRING = 'progress/' + time.strftime("%Y%m%d%H_%M")

colors = cm.rainbow(np.linspace(0, 1, POPULATION_SIZE))
controller_nums = np.arange(start=0,stop=POPULATION_SIZE,step=1)
def interleave(arr1,arr2,axis=0):
    final_shape = [i for i in arr1.shape]
    final_shape[axis] = 2*final_shape[axis]
    arr = np.empty(final_shape,dtype=arr1.dtype)
    if axis==0:
        arr[0::2] = arr1
        arr[1::2] = arr2
    else:
        arr[:,0::2] = arr1
        arr[:,1::2] = arr2

    return arr


def save_plot(saved_object_pos, avoid_pos, saved_controller_pos, gen_no):
    for scenario_num in range(NUM_SIMS):
        fig = plt.figure(scenario_num + 1)
        plt.title('Arena')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        saved_controller_pos = np.array(saved_controller_pos)
        for controller_num in controller_nums:
            plt.plot(saved_controller_pos[:, controller_num, scenario_num, 0],
                     saved_controller_pos[:, controller_num, scenario_num, 1], 'o', label='Robot',
                     markersize=2.0,
                     c=colors[controller_num])
        plt.plot(saved_object_pos[scenario_num,0], saved_object_pos[scenario_num,1],
                 'o', color='red', label='Red ball',
                 markersize=20.0)
        plt.plot(avoid_pos[scenario_num, 0], avoid_pos[scenario_num, 1],
                 'o', color='yellow', label='Red ball',
                 markersize=20.0)

        plt.savefig(PLOT_STRING + '/gen_{0}_{1}.png'.format(gen_no, scenario_num))
        plt.close(fig)

def simulate_step(pos, angle, object_pos,avoid_object_pos,LSTM_states, population,prev_commands,pop_size = POPULATION_SIZE):
    #pos.shape = [POP, sims,2]
    pos_reshaped = tf.reshape(pos,[NUM_SIMS*pop_size,2])         #[SIMS*POP,2]
    angle_reshaped = tf.reshape(angle,[NUM_SIMS*pop_size])

    pos_reordered = tf.stack([pos_reshaped[start_point:None:NUM_SIMS,:] for start_point in range(NUM_SIMS)],axis=0)       #[SIMS,POP,2]
    angle_reordered = tf.stack([angle_reshaped[start_point:None:NUM_SIMS] for start_point in range(NUM_SIMS)],axis=0)

    x,y = tf.unstack(pos_reordered,axis=2)                    #[SIMS,POP,1]

    imgs = Simulator.request.request_images(object_pos, avoid_object_pos, x, y,
                                            angle_reordered)  ##latent_vis shape = [NUM_SIMS, POP, LATENT_DIM]

    imgs = imgs.astype(np.float32)
    imgs = tf.convert_to_tensor(imgs/127.5 - 1.0)
    imgs = tf.reshape(imgs, [NUM_SIMS * POPULATION_SIZE, HEIGHT, WIDTH, NUM_DIM])

    imgs = generator(imgs)
    imgs = imgs + tf.random.normal(tf.shape(imgs[0]), 0.0, 0.01)
    # im = imgs[0]
    # im = (im*127.5 + 127.5).numpy().astype(np.uint8)
    # plt.figure(1)
    # plt.imshow(im)
    # plt.show()
    hidden_states = LSTM_states[0]
    cell_states = LSTM_states[1]

    controller_outputs,h,c = network.predict(imgs, hidden_states, cell_states, population,POPULATION_SIZE)
    controller_noise = tf.random.uniform([1, NUM_SIMS, 2], -SNN_NOISE_SCALE, SNN_NOISE_SCALE)
    reshaped_controller_outputs = tf.reshape(controller_outputs[:, :, 0:2] + controller_noise, [pop_size * NUM_SIMS, 2])

    rand_mag = tf.random.uniform([1], 0.8, 0.9)
    state_changes = motion_sim(reshaped_controller_outputs)
    local_x_change, local_y_change, local_angle_change = tf.unstack(state_changes, axis=1)
    local_x_change = rand_mag * local_x_change * tf.random.uniform([1], 0.97, 1.03)
    local_y_change = rand_mag * local_y_change * tf.random.uniform([1], 0.97, 1.03)
    local_angle_change = local_angle_change * 180.0 * tf.random.uniform([1], 0.97, 1.03)

    local_dist_change = tf.sqrt(tf.square(local_x_change) + tf.square(local_y_change))
    bearing_change = (-tf.math.atan2(local_y_change,local_x_change) + (math.pi/2)) % (math.pi*2.0)

    temp_bearing_c = tf.cast((tf.reshape(angle,shape=[pop_size*NUM_SIMS])/180.0 * math.pi),dtype=tf.float32) + bearing_change
    global_x_change = local_dist_change * tf.sin(temp_bearing_c)
    global_y_change = local_dist_change * tf.cos(temp_bearing_c)

    local_angle_change = tf.reshape(local_angle_change,shape=[pop_size, NUM_SIMS])
    global_x_change = tf.reshape(global_x_change, [pop_size, NUM_SIMS])
    global_y_change = tf.reshape(global_y_change, [pop_size, NUM_SIMS])

    new_global_orientation = (angle + local_angle_change) % 360
    new_global_position_change = tf.stack([global_x_change, global_y_change], axis=2) + pos

    return new_global_position_change, new_global_orientation, LSTM_states, controller_outputs
#tf.concat([reordered_latent_params,reshaped_controller_outputs],axis=1) action + params
def train(saved_string = None):
    gene_shape = [POPULATION_SIZE, network.gene_count]
    if saved_string is None:
        print('Initializing population')
        age = np.zeros([POPULATION_SIZE])
        population = network.initialization()
        gen_no = 0
    else:
        checkpoint = np.load('checkpoints/'+saved_string,allow_pickle=True)
        gen_no = checkpoint['gen_no']
        population = checkpoint['population'].copy()
        age = checkpoint['age'].copy()
        checkpoint = None
        print('Successfully loaded population... Starting at {0}'.format(gen_no))
    half_size = int(NUM_SIMS/2)
    common.mkdir(PLOT_STRING)

    print('Network shape is {0} weights'.format(gene_shape[1]))
    while (gen_no < NUM_EVOLUTIONS):
        print('Gen ' + str(gen_no))
        global MUTATION_SIZE, MUTATION_PROB, NUM_SCENES, TOURNAMENT_SIZE
        if gen_no > (0.9 * NUM_EVOLUTIONS):
            MUTATION_SIZE = 0.01
            MUTATION_PROB = 0.2
        if gen_no > 0.5*NUM_EVOLUTIONS:
            TOURNAMENT_SIZE = 10
        if gen_no == NUM_EVOLUTIONS-1:
            NUM_SCENES = 300

        ##test on every simulator
        start = time.time()
        w = controller.network.get_weights(population)
        fitness = np.zeros([POPULATION_SIZE])
        for batch in range(NUM_SCENES//NUM_SIMS):
            pos = np.random.uniform(MOVABLE_BOARD_RANGE_X[0],MOVABLE_BOARD_RANGE_X[1],[1, half_size, 2])
            angle = np.random.uniform(0,360.0,[1, half_size])

            object_pos_x = np.random.uniform(MOVABLE_BOARD_RANGE_X[0],MOVABLE_BOARD_RANGE_X[1],[half_size,1])
            object_pos_y = np.random.uniform(0.3,MOVABLE_BOARD_RANGE_Y[1],[half_size,1])

            object_pos = np.concatenate([object_pos_x,object_pos_y],axis=1)

            avoid_object_pos_x = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [half_size, 1])
            avoid_object_pos_y = np.random.uniform(MOVABLE_BOARD_RANGE_Y[0], -0.3, [half_size, 1])
            avoid_object_pos = np.concatenate([avoid_object_pos_x,avoid_object_pos_y],axis=1)


            i = 0
            while i < half_size:
                if (np.linalg.norm(pos[0,i] - object_pos[i],axis=0) < 0.25) or (np.linalg.norm(pos[0,i] - avoid_object_pos[i],axis=0) < 0.25):
                    pos[0,i] = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [2])
                else:
                    i += 1

            i = 0
            while i < 2:
                if (np.linalg.norm(pos[0, i] - object_pos[i], axis=0) < 1):
                    pos[0, i] = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [2])
                else:
                    i += 1


            object_pos2 = avoid_object_pos.copy()
            avoid_object_pos2 = object_pos.copy()
            pos = interleave(pos,pos.copy(),axis=1)
            angle = interleave(angle,angle.copy(),axis=1)
            object_pos = interleave(object_pos,object_pos2)
            avoid_object_pos = interleave(avoid_object_pos,avoid_object_pos2)


            pos = np.tile(pos,[POPULATION_SIZE,1,1])
            angle = np.tile(angle,[POPULATION_SIZE,1])

            #initialize object params for gen
            saved_controller_pos = []
            LSTM_states = list(network.generate_first_hidden_states())
            prev_commands = tf.zeros(shape=[POPULATION_SIZE,NUM_SIMS,2])
            for time_step in range(MAX_TIME_STEPS):
                pos, angle, LSTM_states, prev_commands = simulate_step(pos,angle,object_pos,avoid_object_pos,LSTM_states,w,prev_commands)
                saved_controller_pos.append(pos)



            dist_from_target = np.zeros(shape=[MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])
            dist_from_avoid = np.zeros(shape=[MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])
            dist_over_boarder = np.zeros(shape=[MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])
            has_reached = np.zeros(shape = [MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])
            min_steps_to_reach = np.ones([POPULATION_SIZE,NUM_SIMS])*(MAX_TIME_STEPS+1)
            fitness_i = np.zeros([POPULATION_SIZE,NUM_SIMS])
            pos_shp = [POPULATION_SIZE,NUM_SIMS,2]
            for t in range(MAX_TIME_STEPS):
                dist_from_target[t, :, :] = np.linalg.norm(object_pos - saved_controller_pos[t], axis=2)
                min_steps_to_reach = np.minimum(min_steps_to_reach,np.where(dist_from_target[t] < CLOSE_ENOUGH_TO_STOP,t,MAX_TIME_STEPS+1))

            closest_dist_from_target = np.amin(dist_from_target, axis=0)
            closest_dist_from_target = np.where(closest_dist_from_target < CLOSE_ENOUGH_TO_STOP, CLOSE_ENOUGH_TO_STOP,
                                                closest_dist_from_target)

            divident = (MAX_TIME_STEPS - min_steps_to_reach) / MAX_TIME_STEPS
            fitness_i = np.where(min_steps_to_reach < MAX_TIME_STEPS + 1, divident, -1.0 * closest_dist_from_target)
            fitness_i = np.sum(fitness_i, axis=1)
            fitness = fitness + fitness_i

            if gen_no % SAVE_EVERY == 0 and batch == 0:
                save_plot(object_pos, avoid_object_pos, saved_controller_pos, gen_no)
        avg_fitness = np.sum(fitness, axis=0)



        new_population = []
        new_age = np.zeros_like(age)
        ##selection
        parents = []
        cur_best_value = -math.inf
        cur_best = None
        cur_best_age = 0
        new_age_index = 0
        #tournament selection
        for i in range(int(POPULATION_SIZE/TOURNAMENT_SIZE)):
            tournament_winner_i = np.argmax(fitness[i*TOURNAMENT_SIZE:(i+1)*TOURNAMENT_SIZE])
            parents.append(population[(i * TOURNAMENT_SIZE) + tournament_winner_i])



            if fitness[(i * TOURNAMENT_SIZE) + tournament_winner_i] > cur_best_value:
                cur_best_value = fitness[(i * TOURNAMENT_SIZE) + tournament_winner_i]
                cur_best = population[(i * TOURNAMENT_SIZE) + tournament_winner_i]
                cur_best_age = age[(i * TOURNAMENT_SIZE) + tournament_winner_i]

            if ELITISM:
                new_population.append(population[(i * TOURNAMENT_SIZE) + tournament_winner_i])
                new_age[new_age_index] = age[(i * TOURNAMENT_SIZE) + tournament_winner_i]
                new_age_index+=1

        if gen_no > 0 and gen_no % HEADLESS == 0 and gen_no < (0.9*NUM_EVOLUTIONS):
            hc = network.initialization()
            for i in range(int((POPULATION_SIZE - len(parents)) if ELITISM else POPULATION_SIZE)):
                parent1 = parents[np.random.randint(0, POPULATION_SIZE / TOURNAMENT_SIZE)]

                parent2 = hc[np.random.randint(0, POPULATION_SIZE)]

                child = np.where(np.random.uniform(0.0, 1.0, parent1.shape) <= 0.8, parent1, parent2)
                new_population.append(child)
                new_age[new_age_index] = gen_no
                new_age_index += 1

        else:
            ##uniform_crossover
            for i in range(int((POPULATION_SIZE-len(parents)) if ELITISM else POPULATION_SIZE)):
                parent1 = parents[np.random.randint(0,POPULATION_SIZE/TOURNAMENT_SIZE)]

                parent2 = parent1
                while np.array_equal(parent1,parent2):
                    parent2 = parents[np.random.randint(0,POPULATION_SIZE/TOURNAMENT_SIZE)]

                child = np.where(np.random.uniform(0.0,1.0,parent1.shape) <= 0.5, parent1, parent2)
                new_population.append(child)
                new_age[new_age_index] = gen_no
                new_age_index +=1
        #mutation

        no_mutation = np.ones([int(POPULATION_SIZE/TOURNAMENT_SIZE),population.shape[1]])
        mutation = np.random.uniform(0.0,1.0,population[int(POPULATION_SIZE/TOURNAMENT_SIZE):None].shape if ELITISM else population.shape)

        mask = np.concatenate((no_mutation,mutation),axis=0)


        mutation_mask = mask <= MUTATION_PROB

        mutation = np.where(mutation_mask,np.random.normal(0,MUTATION_SIZE,population.shape),np.zeros(population.shape))

        new_population += mutation




        if gen_no%SAVE_CONTROLLER_EVERY == 0 or gen_no == NUM_EVOLUTIONS-1:
            np.save(SAVE_STRING + 'gen_no-{0}fitness-{1:.2f}pop{2:.2f}.npy'.format(gen_no,cur_best_value,avg_fitness/POPULATION_SIZE),cur_best,allow_pickle=True)

        if gen_no%CHECKPOINT_EVERY == 0 or gen_no == NUM_EVOLUTIONS-1:
            for file in os.listdir('checkpoints'):
                os.remove(os.path.join('checkpoints', file))
            np.savez('checkpoints/saved_{0}_{1}'.format(gen_no,BETA),gen_no = gen_no,population = population,age = age)

        population = new_population
        age = new_age


        gen_no += 1
        indices = np.arange(len(population))
        np.random.shuffle(indices)

        population = population[indices]
        age = age[indices]
        end = time.time()
        print(MUTATION_SIZE)
        print('Time taken: {0}\nAverage fitness = {1:.3f} \nBest candidate fitness: {2:.3f}\nWinner age: {3}'.format(end-start,avg_fitness/POPULATION_SIZE,cur_best_value,cur_best_age))

    # gauntlet()
    # for file in os.listdir('checkpoints'):
    #     os.remove(os.path.join('checkpoints', file))


if __name__ == "__main__":
    try:
        print(tf.test.is_built_with_cuda())
        gpus = tf.config.list_physical_devices('GPU')
        list_dir = os.listdir('checkpoints')
        if len(list_dir) == 0:
            # for i in range(2):
            train()
        else:
            file = list_dir[0].title()
            train(file)
        # gauntlet()
    except KeyboardInterrupt:
        pass
    Simulator.request.end()




