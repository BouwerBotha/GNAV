import time

import numpy as np
import tensorflow as tf
from common import POPULATION_SIZE,NUM_SIMS, INCL_PREV_COMMANDS,HEIGHT,WIDTH
##HYPER_PARAMS

POSITIONAL_ENCODING = True
dim_q = 8
NUM_MOST_IMPORTANT = 6
NUM_HEADS = 1
##


PATCH_SIZE = 7
PATCH_STRIDE = 3



n1 = int((HEIGHT-PATCH_SIZE)/PATCH_STRIDE + 1)
n2 = int((WIDTH-PATCH_SIZE)/PATCH_STRIDE + 1)
NUM_PATCHES = n1*n2
offset = PATCH_SIZE//2
PATCH_CENTERS = []
for i in range(n1):
    patch_center_row = offset + i * PATCH_STRIDE
    for j in range(n2):
        patch_center_col = offset + j * PATCH_STRIDE
        PATCH_CENTERS.append([patch_center_row, patch_center_col])
patch_centers = tf.convert_to_tensor(PATCH_CENTERS)

positional_encoding = tf.cast(patch_centers,dtype=tf.float32)
positional_encoding = tf.stack([positional_encoding[:,0]/HEIGHT,positional_encoding[:,1]/WIDTH],axis=-1)
positional_encoding = tf.expand_dims(positional_encoding,axis=0)
positional_encoding = tf.expand_dims(positional_encoding,axis=0)
positional_encoding = tf.tile(positional_encoding,[NUM_SIMS,POPULATION_SIZE,1,1])
positional_encoding = tf.expand_dims(positional_encoding,axis=2)

GRU = True
cells = [30]
nn_output_dims = 2

###load weight shapes
weight_shapes = []

weight_shapes.append([POPULATION_SIZE,NUM_HEADS,((PATCH_SIZE**2) * 3) + 2 if POSITIONAL_ENCODING else ((PATCH_SIZE**2) * 3),dim_q]) #key
weight_shapes.append([POPULATION_SIZE,NUM_HEADS,((PATCH_SIZE**2) * 3) + 2 if POSITIONAL_ENCODING else ((PATCH_SIZE**2) * 3),dim_q]) #query

for i in range(len(cells) + 1): ##LSTM layer + output layer
    if i == 0:
        input_size = (NUM_MOST_IMPORTANT*2*NUM_HEADS) ##row and col info for imp point
    else:
        input_size = cells[i - 1]

    not_output = i < len(cells)

    if not_output:
        num_cells = cells[i]
        if not GRU:
            for k in range(4):      #for each gate
                weight_shapes.append([POPULATION_SIZE, input_size, num_cells])
                weight_shapes.append([POPULATION_SIZE, num_cells, num_cells])
                weight_shapes.append([POPULATION_SIZE, 1, num_cells])
        else:
            for k in range(3):      #for each gate
                weight_shapes.append([POPULATION_SIZE, input_size, num_cells])
                weight_shapes.append([POPULATION_SIZE, num_cells, num_cells])
                weight_shapes.append([POPULATION_SIZE, 1, num_cells])

    else:
        num_cells = 2
        weight_shapes.append([POPULATION_SIZE, input_size, num_cells])  # for fc layer
        weight_shapes.append([POPULATION_SIZE, 1, num_cells])



gene_count = 0

for w in weight_shapes:
    gene_count += np.product(w[1:None])

#glorot uniform [-sqrt(6/(in + out)); sqrt(6/(in + out))]
def initialization():
    weights = np.zeros([POPULATION_SIZE,gene_count])

    ##attention
    count = 0
    for w in weight_shapes:
        in_plus_out = np.sum(w[1:None])
        in_times_out = np.product(w[1:None])
        weights[:,count:count+in_times_out] = np.random.uniform(-np.sqrt(6.0/in_plus_out),np.sqrt(6.0/in_plus_out),[POPULATION_SIZE,in_times_out])
        count+=in_times_out

    return weights

def generate_first_hidden_states(training = True,num_scenes = 1,gauntlet_size = 0):

    if not GRU:
        hidden_states = []
        cell_states = []
        for i in range(len(cells)):
            if gauntlet_size > 0:
                hidden_states.append(tf.zeros(shape=[gauntlet_size, NUM_SIMS, cells[i]]))
                cell_states.append(tf.zeros(shape=[gauntlet_size, NUM_SIMS, cells[i]]))
            elif training:
                hidden_states.append(tf.zeros(shape=[POPULATION_SIZE, NUM_SIMS, cells[i]]))
                cell_states.append(tf.zeros(shape=[POPULATION_SIZE, NUM_SIMS, cells[i]]))
            else:
                hidden_states.append(tf.zeros(shape=[num_scenes, cells[i]]))
                cell_states.append(tf.zeros(shape=[num_scenes, cells[i]]))

        return hidden_states, cell_states
    else:
        hidden_states = []
        for i in range(len(cells)):
            if gauntlet_size > 0:
                hidden_states.append(tf.zeros(shape=[gauntlet_size, NUM_SIMS, cells[i]]))
            elif training:
                hidden_states.append(tf.zeros(shape=[POPULATION_SIZE, NUM_SIMS, cells[i]]))
            else:
                hidden_states.append(tf.zeros(shape=[num_scenes, cells[i]]))

        return hidden_states, []



def get_weights(population_genes):
    weights = []
    count = 0
    for i in range(len(weight_shapes)):

        current_w = weight_shapes[i]

        if population_genes.shape[0] != POPULATION_SIZE:
            current_w[0] = population_genes.shape[0]

        w_size = np.product(current_w[1:None])

        weights.append(tf.cast(tf.convert_to_tensor(population_genes[:,count:count+w_size].reshape(current_w)),dtype=tf.float32))

        count += w_size

    if gene_count != count:
        print("Error {0}:{1}".format(gene_count,count))

    return weights


def get_single_network_weights(controller_genes):
    weights = []
    count = 0

    for i in range(len(weight_shapes)):
        current_weight = weight_shapes[i]

        weight_size = np.product(current_weight[1:None])

        weights.append(controller_genes[count:count+weight_size].reshape(current_weight[1:None]))

        count+= weight_size

    if count != gene_count:
        print('Error loading weights')

    return weights




def predict(inputs, hidden_states, cell_states,layer_weights,POPULATION_SIZE):
    weight_index = 0
    input = None
    ##inputs shape [POP*NUM_SIMS,W,H,3]
    ##generate patches
    patches = tf.image.extract_patches(inputs,sizes=[1,PATCH_SIZE,PATCH_SIZE,1],strides=[1,PATCH_STRIDE,PATCH_STRIDE,1],rates=[1,1,1,1],padding='VALID')
    input = tf.reshape(patches,[NUM_SIMS,POPULATION_SIZE,n2,n1,-1])
    input = tf.reshape(input, [NUM_SIMS,POPULATION_SIZE,NUM_PATCHES,tf.shape(input)[-1]])
    input = tf.expand_dims(input,axis=2)
    if POSITIONAL_ENCODING:
        input = tf.concat([input,positional_encoding[:,0:POPULATION_SIZE]],axis=-1)
    K = tf.matmul(input, layer_weights[weight_index])
    Q = tf.matmul(input,layer_weights[weight_index+1])
    input = None
    scaled_attention_logits = tf.matmul(K, tf.transpose(Q, perm=[0, 1, 2,4, 3])) / tf.sqrt(NUM_PATCHES * 1.0)
    weight_index+=2
    patch_importance = tf.nn.softmax(scaled_attention_logits,axis=-1)
    K = None
    Q = None
    scaled_attention_logits = None
    patch_importance_sum = tf.reduce_sum(patch_importance,axis=-2)
    ix = tf.argsort(patch_importance_sum,axis=-1,direction='DESCENDING')
    input = None
    patches = None
    top_k_ix = ix[:,:,:,0:NUM_MOST_IMPORTANT]
    top_k_ix_flattened = tf.reshape(top_k_ix,[-1])
    centers = tf.gather(patch_centers,top_k_ix_flattened)
    old_shape = tf.shape(top_k_ix)
    centers = tf.reshape(centers,shape=[old_shape[0]*old_shape[1],old_shape[2],old_shape[3],2])
    reordered_centers = tf.stack([centers[start_point:None:old_shape[1]] for start_point in range(old_shape[1])],axis=0)
    reordered_centers = tf.concat([reordered_centers[:,:,i] for i in range(NUM_HEADS)],axis=2)
    reordered_centers = tf.cast(tf.stack([reordered_centers[:,:,:,0]/HEIGHT,reordered_centers[:,:,:,1]/WIDTH],axis=-1),tf.float32)
    reshaped_centers = tf.reshape(reordered_centers,[old_shape[1],old_shape[0],-1])
    input = reshaped_centers
    for i in range(len(cells) + 1): #LSTM layers + outputlayer

        if i < len(cells):         #LSTM or GRU layer
            if not GRU:
                f_t = tf.nn.sigmoid(tf.matmul(input,layer_weights[weight_index]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 1]) + layer_weights[weight_index + 2])
                i_t = tf.nn.sigmoid(tf.matmul(input,layer_weights[weight_index+3]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 4]) + layer_weights[weight_index + 5])
                o_t = tf.nn.sigmoid(tf.matmul(input,layer_weights[weight_index+6]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 7])+ layer_weights[weight_index+8])
                c_t = tf.nn.tanh(tf.matmul(input,layer_weights[weight_index+9]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 10]) + layer_weights[weight_index+11])
                weight_index += 12

                cell_states[i] = tf.multiply(f_t,cell_states[i]) + tf.multiply(i_t,c_t)
                hidden_states[i] = tf.multiply(o_t,tf.nn.tanh(cell_states[i]))

                input = hidden_states[i]
            else:
                z_t = tf.nn.sigmoid(tf.matmul(input,layer_weights[weight_index]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 1]) + layer_weights[weight_index + 2])
                r_t = tf.nn.sigmoid(tf.matmul(input,layer_weights[weight_index+3]) + tf.matmul(hidden_states[i],layer_weights[weight_index + 4]) + layer_weights[weight_index + 5])
                h_t_hat = tf.nn.tanh(tf.matmul(input,layer_weights[weight_index+6]) + tf.multiply(r_t,tf.matmul(hidden_states[i],layer_weights[weight_index + 7]))+ layer_weights[weight_index+8])
                hidden_states[i] = tf.multiply(z_t,hidden_states[i]) + tf.multiply(1-z_t,h_t_hat)

                weight_index += 9

                input = hidden_states[i]
            # # 50 fully connected cells
            # input = tf.nn.tanh(tf.matmul(input,layer_weights[weight_index]) + layer_weights[weight_index+1])
            # weight_index += 2

        else:
            #if moving backwards
            # input = 0.9 * tf.nn.tanh(tf.matmul(input, layer_weights[weight_index]) + layer_weights[weight_index+1]) + 0.1
            input = 0.9*tf.nn.tanh(tf.matmul(input, layer_weights[weight_index]) + layer_weights[weight_index+1]) +0.1
            weight_index += 2
    if POPULATION_SIZE == 1:
        return input, hidden_states, cell_states, centers
    else:
        return input, hidden_states, cell_states