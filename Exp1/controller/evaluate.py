import numpy as np
import tensorflow as tf
import math
import SNN.network
import matplotlib.pyplot as plt
import common
import controller.network
from common import  MOVABLE_BOARD_RANGE_X, MOVABLE_BOARD_RANGE_Y, CLOSE_ENOUGH_TO_STOP, MAX_TIME_STEPS, HEIGHT,WIDTH,NUM_DIM,NUM_EVAL_TIMES, EVAL_MAX_TIME_STEPS, INCL_PREV_COMMANDS, COMMAND_STEP_SKIP
import Simulator.request
from PIL import Image
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

NUM_SIMS = 10
STEP_INTERVAL = 1
half_size = int(NUM_SIMS/2)

HEAD_COLORS = [np.array([0,0,255]),np.array([0,255,255]),np.array([255,0,255])]

max_x_velocity = (MOVABLE_BOARD_RANGE_X[1] - MOVABLE_BOARD_RANGE_X[0]) / MAX_TIME_STEPS
max_y_velocity = 0

SNN_PATH = '../SNN/checkpoints/snn.ckpt'
CONTROLLER_STRING = 'saved_controllers/pictures_GRU_30_1_atthead_IMPORTANCE_Patch_SIZE_7gen_no-0fitness--3.00pop-6.90.npy'

motion_sim = SNN.network.get_model()
motion_sim.load_weights(SNN_PATH)




def reverse_parse(images):
    return tf.cast(tf.maximum(0.0, tf.minimum(255.0, ((images + 1) / 2) * 255.0)), tf.uint8)
    # return tf.cast(tf.maximum(0.0, tf.minimum(255.0, (images * 255.0))), tf.uint8)

def get_sign(bearing):
    if bearing <= 90:
        return (1,1)
    elif bearing <= 180:
        return (-1,1)
    elif bearing <= 270:
        return (-1,-1)
    else:
        return (1,-1)       #sin, cos

def get_bearing(bearing):
    if bearing <= 90:
        return 90 - bearing
    elif bearing <= 180:
        return bearing - 90
    elif bearing <= 270:
        return 90 - (bearing - 180)
    else:
        return bearing - 270


def simulate_step(time_step,pos, angle, object_pos,avoid_object_pos, LSTM_states, population,prev_commands):
    #pos.shape = [POP, sims,2]
    POPULATION_SIZE = 1
    # pos.shape = [POP, sims,2]
    pos_reshaped = tf.reshape(pos, [NUM_SIMS * POPULATION_SIZE, 2])  # [SIMS*POP,2]
    angle_reshaped = tf.reshape(angle, [NUM_SIMS * POPULATION_SIZE])

    pos_reordered = tf.stack([pos_reshaped[start_point:None:NUM_SIMS, :] for start_point in range(NUM_SIMS)],
                             axis=0)  # [SIMS,POP,2]
    angle_reordered = tf.stack([angle_reshaped[start_point:None:NUM_SIMS] for start_point in range(NUM_SIMS)], axis=0)

    x, y = tf.unstack(pos_reordered, axis=2)  # [SIMS,POP,1]

    imgs = Simulator.request.request_images(object_pos,avoid_object_pos, x, y,
                                            angle_reordered,NUM_SIMS)  ##latent_vis shape = [NUM_SIMS, POP, LATENT_DIM]



    image_float = imgs.astype(np.float32)
    image_float = tf.convert_to_tensor(image_float / 127.5) - 1.0
    reshaped_images = tf.reshape(image_float, [NUM_SIMS * POPULATION_SIZE, HEIGHT, WIDTH, NUM_DIM])
    reshaped_images = generator(reshaped_images)
    hidden_states = LSTM_states[0]
    cell_states = LSTM_states[1]

    controller_outputs, h, c, reshaped_centers = controller.network.predict(reshaped_images, hidden_states, cell_states, population, POPULATION_SIZE)

    reshaped_controller_outputs = tf.reshape(controller_outputs, [POPULATION_SIZE * NUM_SIMS, 2])

    state_changes = motion_sim(reshaped_controller_outputs)

    local_x_change, local_y_change, local_angle_change = tf.unstack(state_changes, axis=1)
    local_x_change = 0.82 * local_x_change
    local_y_change = 0.82 * local_y_change
    local_angle_change = local_angle_change * 180.0

    local_dist_change = tf.sqrt(tf.square(local_x_change) + tf.square(local_y_change))
    bearing_change = (-tf.math.atan2(local_y_change, local_x_change) + (math.pi / 2)) % (math.pi * 2.0)

    temp_bearing_c = tf.cast((tf.reshape(angle, shape=[POPULATION_SIZE * NUM_SIMS]) / 180.0 * math.pi),
                             dtype=tf.float32) + bearing_change
    global_x_change = local_dist_change * tf.sin(temp_bearing_c)
    global_y_change = local_dist_change * tf.cos(temp_bearing_c)

    local_angle_change = tf.reshape(local_angle_change, shape=[POPULATION_SIZE, NUM_SIMS])
    global_x_change = tf.reshape(global_x_change, [POPULATION_SIZE, NUM_SIMS])
    global_y_change = tf.reshape(global_y_change, [POPULATION_SIZE, NUM_SIMS])

    new_global_orientation = (angle + local_angle_change) % 360
    new_global_position_change = tf.stack([global_x_change, global_y_change], axis=2) + pos

    return new_global_position_change, new_global_orientation, LSTM_states, controller_outputs, reshaped_images, reshaped_centers

def evaluate(weights):
    POPULATION_SIZE = 1
    w = controller.network.get_single_network_weights(weights)

    pos = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [1, half_size, 2])
    angle = np.random.uniform(0, 360.0, [1, half_size])

    object_pos_x = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [half_size, 1])
    object_pos_y = np.random.uniform(0.3, MOVABLE_BOARD_RANGE_Y[1], [half_size, 1])

    object_pos = np.concatenate([object_pos_x, object_pos_y], axis=1)

    avoid_object_pos_x = np.random.uniform(MOVABLE_BOARD_RANGE_X[0], MOVABLE_BOARD_RANGE_X[1], [half_size, 1])
    avoid_object_pos_y = np.random.uniform(MOVABLE_BOARD_RANGE_Y[0], -0.3, [half_size, 1])

    avoid_object_pos = np.concatenate([avoid_object_pos_x, avoid_object_pos_y], axis=1)

    pos = np.concatenate([pos, pos.copy()], axis=1)
    angle = np.concatenate([angle, angle.copy()], axis=1)
    object_pos2 = avoid_object_pos.copy()
    avoid_object_pos2 = object_pos.copy()
    object_pos = np.concatenate([object_pos, object_pos2], axis=0)
    avoid_object_pos = np.concatenate([avoid_object_pos, avoid_object_pos2], axis=0)



    saved_controller_pos = []
    saved_controller_angle = []

    LSTM_states = list(controller.network.generate_first_hidden_states(False,NUM_SIMS))
    prev_commands = tf.zeros([POPULATION_SIZE,NUM_SIMS, 2])
    all_images = []
    all_commands = []
    all_centers = []
    for time_step in range(EVAL_MAX_TIME_STEPS):
        pos, angle, LSTM_states, prev_commands, images, reshaped_centers= simulate_step(time_step,pos, angle, object_pos, avoid_object_pos, LSTM_states, w, prev_commands)
        all_commands.append(prev_commands)
        all_images.append(images)
        all_centers.append(reshaped_centers)
        saved_controller_pos.append(pos)
        saved_controller_angle.append(angle)

    dist_from_target = np.zeros(shape=[EVAL_MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])
    for t in range(EVAL_MAX_TIME_STEPS):
        dist_from_target[t, :, :] = np.linalg.norm(object_pos - saved_controller_pos[t], axis=2)
    closest_dist_from_target = np.amin(dist_from_target, axis=0)
    min_steps_to_reach = np.ones_like(closest_dist_from_target) * -1000.0


    for i in range(EVAL_MAX_TIME_STEPS-1,-1,-1):
        min_steps_to_reach = np.where(dist_from_target[i] < CLOSE_ENOUGH_TO_STOP,i,min_steps_to_reach)

    print(np.amin(min_steps_to_reach,axis=0))
    #visualization
    Simulator.request.end()
    all_images_reversed = []
    for i in range(len(all_images)):
        all_images_reversed.append(reverse_parse(all_images[i]))
    for scenario_num in range(NUM_SIMS):
        fig = plt.figure(scenario_num)
        plt.title('Arena')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        stopped = False
        for time_step in range(0, EVAL_MAX_TIME_STEPS, STEP_INTERVAL):
            # if (math.dist(saved_controller_pos[time_step][0, scenario_num, :],
            #               object_pos[scenario_num, :]) > CLOSE_ENOUGH_TO_STOP) and not stopped:
                plt.plot(object_pos[scenario_num, 0], object_pos[scenario_num, 1],
                         'o', color='red', label='Red ball',
                         markersize=20.0)
                plt.plot(avoid_object_pos[scenario_num, 0], avoid_object_pos[scenario_num, 1],
                         'o', color='yellow', label='Yellow ball',
                         markersize=20.0)
                plt.plot(saved_controller_pos[time_step][0, scenario_num, 0],
                         saved_controller_pos[time_step][0, scenario_num, 1], 'o', label='Robot', markersize=20.0,
                         color=(0, ((time_step * STEP_INTERVAL) + 5) / 255, 0))
                pos_bearing = saved_controller_angle[time_step][0, scenario_num]
                bearing_mag = get_bearing(pos_bearing)
                dx = 0.1 * math.cos(bearing_mag / 180.0 * math.pi) * get_sign(pos_bearing)[1]
                dy = 0.1 * math.sin(bearing_mag / 180.0 * math.pi) * get_sign(pos_bearing)[0]
                plt.arrow(saved_controller_pos[time_step][0, scenario_num, 0],
                          saved_controller_pos[time_step][0, scenario_num, 1], dx, dy,
                          color=(0, ((time_step * STEP_INTERVAL) + 5) / 255, 0))

            # else:
            #     stopped = True

    #
    #
    image_count = 0

    for scenario_num in range(NUM_SIMS):
        fig = plt.figure(NUM_SIMS + scenario_num)
        num_cols = 10
        num_rows = int(EVAL_MAX_TIME_STEPS/num_cols)
        figure, axis = plt.subplots(num_rows, num_cols)
        for i in range(num_rows):
            for k in range(num_cols):
                img = Image.fromarray(all_images_reversed[(i*num_cols)+k][scenario_num].numpy())
                image_count += 1
                axis[i, k].imshow(img)
                axis[i, k].set_title('time t = %d' % (((i * num_rows) + k) * STEP_INTERVAL))
    offset = controller.network.offset
    for scenario_num in range(NUM_SIMS):
        print('Scenario %d' % scenario_num)
        fig = plt.figure(NUM_SIMS + scenario_num+ NUM_SIMS)
        num_cols = 10
        num_rows = int(EVAL_MAX_TIME_STEPS / num_cols)
        figure, axis = plt.subplots(num_rows, num_cols)
        for i in range(num_rows):
            for k in range(num_cols):
                print(all_commands[(i * num_cols) + k][0, scenario_num])
                img = all_images_reversed[(i * num_cols) + k][scenario_num].numpy()
                img_new = np.zeros_like(img)
                all_centers[(i * num_cols) + k] = np.reshape(all_centers[(i * num_cols) + k],[NUM_SIMS,controller.network.NUM_HEADS,controller.network.NUM_MOST_IMPORTANT,2])
                for att in range(controller.network.NUM_HEADS):
                    color = HEAD_COLORS[att]
                    for num in range(controller.network.NUM_MOST_IMPORTANT):
                        ##extract filter

                        cur_center = all_centers[(i * num_cols) + k][scenario_num,att,num]
                        row = cur_center[0]
                        col = cur_center[1]
                        cur_filter = img[row - offset:row + offset, col - offset:col + offset]
                        img_new[row - offset:row + offset, col - offset:col + offset] = cur_filter
                        img_new[row + (offset + 2):row + (offset + 1), col - (offset + 1):col + (offset + 1)] = color
                        img_new[row - (offset + 2):row - (offset + 1), col - (offset + 1):col + (offset + 1)] = color
                        img_new[row - (offset + 1):row + (offset + 1), col + (offset + 2):col + (offset + 1)] = color
                        img_new[row - (offset + 1):row + (offset + 1), col - (offset + 2):col - (offset + 1)] = color
                axis[i, k].imshow(Image.fromarray(img_new))
                axis[i, k].set_title('time t = %d' % (((i * num_cols) + k) * STEP_INTERVAL))

    plt.show()


def evaluate_saved(save_string,weights,exp_str):
    w = controller.network.get_single_network_weights(weights)
    container = np.load(save_string,allow_pickle=True)
    POPULATION_SIZE = 1
    global_start_pos = container['global_start_pos']
    global_start_angle = container['global_start_angle']

    global_saved_controller_pos = []
    global_saved_controller_angle = []
    global_saved_object_pos = list(container['saved_object_pos'])
    min_steps_to_reach_all = np.zeros([global_start_pos.shape[0]])
    num_times = int(global_start_pos.shape[0]/NUM_SIMS)
    for i in range(num_times):
        saved_controller_pos = []
        saved_controller_angle = []
        saved_object_pos = []
        pos = global_start_pos[(i*NUM_SIMS):(i+1)*NUM_SIMS]
        angle = global_start_angle[(i*NUM_SIMS):(i+1)*NUM_SIMS]
        LSTM_states = list(controller.network.generate_first_hidden_states(False, NUM_SIMS))
        prev_commands = tf.zeros([POPULATION_SIZE,NUM_SIMS, 2])
        all_latents = []
        all_commands = []
        object_pos = global_saved_object_pos[0][(i*NUM_SIMS):(i+1)*NUM_SIMS]
        for time_step in range(EVAL_MAX_TIME_STEPS):
            pos, angle, LSTM_states, prev_commands, latents = simulate_step(time_step,pos, angle, object_pos,LSTM_states, w, prev_commands)
            all_commands.append(prev_commands)
            all_latents.append(latents)
            saved_controller_pos.append(pos)
            saved_controller_angle.append(angle)
            object_pos = global_saved_object_pos[time_step+1][(i*NUM_SIMS):(i+1)*NUM_SIMS]
            saved_object_pos.append(object_pos)

        dist_from_target = np.zeros(shape=[EVAL_MAX_TIME_STEPS, POPULATION_SIZE, NUM_SIMS])

        for t in range(EVAL_MAX_TIME_STEPS):
            dist_from_target[t, :, :] = np.linalg.norm(saved_object_pos[t] - saved_controller_pos[t], axis=2)

        closest_dist_from_target = np.amin(dist_from_target, axis=0)
        min_steps_to_reach = np.ones_like(closest_dist_from_target) * -1000.0

        for k in range(EVAL_MAX_TIME_STEPS - 1, -1, -1):
            min_steps_to_reach = np.where(dist_from_target[k] < CLOSE_ENOUGH_TO_STOP, k, min_steps_to_reach)


        min_steps_to_reach_all[(i * NUM_SIMS): (i + 1) * NUM_SIMS] = min_steps_to_reach[0]
        global_saved_controller_pos.append(saved_controller_pos)
        global_saved_controller_angle.append(saved_controller_angle)

        print('Done with %d' %((i+1)*NUM_SIMS))
    print(min_steps_to_reach_all)

    mixed_pos = np.array(global_saved_controller_pos)

    pos = np.concatenate([elem for elem in mixed_pos],axis=1)

    mixed_angle = np.array(global_saved_controller_angle)

    angle = np.concatenate([elem for elem in mixed_angle],axis=1)
    print('saving results')
    np.savez(exp_str,pos=pos,angle=angle, min_steps = min_steps_to_reach_all)

if __name__ == '__main__':
    try:
        weights = np.load(CONTROLLER_STRING,allow_pickle=True)
        save_str = '../evaluation/scenario1.npz'
        evaluate(weights)
        # for i in range(NUM_EVAL_TIMES):
        #     print('Evaluation: %d' %(i+1))
        #     export_string = 'baseresults2layers_{0}.npz'.format(i)
        #     evaluate_saved(save_str,weights,export_string)
    except KeyboardInterrupt:
        pass
    Simulator.request.end()