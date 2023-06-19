import numpy as np
import tensorflow as tf

VALIDATION_PERCENTAGE = 0.15
TEST_PERC = 0.1

def get_data():
    txt = np.loadtxt('../data/localised_movements.csv')
    x_data = np.zeros(shape=[txt.shape[0],2])
    x_data[:,0] = txt[:,0] #left
    x_data[:,1] = txt[:,1] #right
    x_data = x_data.astype('float32')
    y_data = np.zeros(shape=[txt.shape[0],3])
    y_data[:,0] = txt[:,2]      #x_change
    y_data[:,1] = txt[:,3]      #y_change
    y_data[:,2] = txt[:,4]/180.0      #theta_change
    y_data = y_data.astype('float32')
    training_size = int((1-TEST_PERC-VALIDATION_PERCENTAGE)*x_data.shape[0])
    validation_size = int(VALIDATION_PERCENTAGE*x_data.shape[0])
    test_size = x_data.shape[0]-training_size-validation_size

    ds_full = tf.data.Dataset.from_tensor_slices((x_data,y_data))
    ds_full = ds_full.shuffle(x_data.shape[0])
    ds_train = ds_full.take(training_size)
    ds_train = ds_train.batch(32)

    ds_val = ds_full.skip(training_size).take(validation_size)
    ds_val = ds_val.batch(validation_size)
    ds_test = ds_full.skip(training_size+validation_size).skip(test_size)
    ds_test = ds_test.batch(test_size)

    return ds_train, ds_val, ds_test
