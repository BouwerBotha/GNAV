import tensorflow as tf
from SNN import data, network
import keras_tuner
from keras.callbacks import ModelCheckpoint,EarlyStopping
ds_train, ds_val, ds_test = data.get_data()
CHECKPOINT_PATH = 'checkpoints/snn.ckpt'

early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 10, mode = "min")
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=5,factor = 0.2,mode='min')
def optimize_hyperparams():
    # tuner = keras_tuner.tuners.Hyperband(
    #     network.build_model,
    #     objective='val_loss',
    #     max_epochs=1000,  # max epochs per model
    #     executions_per_trial=4)  # number of models built for each hyper conf
    tuner = keras_tuner.RandomSearch(network.build_model,objective='val_loss',max_trials=30)
    tuner.search(ds_train, validation_data=ds_val, epochs=500, callbacks=[early_stop,reduce_lr])

    models = tuner.get_best_models()

    tuner.results_summary()




def train_optimized_network():
    NUM_TRIES = 10
    best_loss = 1e10

    cur_best_model = None

    for i in range(NUM_TRIES):
        model = network.get_model()

        CHECKPOINT_PATH = 'checkpoints/snn.ckpt'
        cp_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH, save_best_only=True, save_weights_only=True, verbose=0,
                                      monitor='val_loss', mode="min")
        history = model.fit(ds_train, validation_data=ds_val, epochs=1000, callbacks=[early_stop, cp_callback,reduce_lr])
        val_losses = history.history['val_loss']
        val_loss = min(val_losses)
        if val_loss < best_loss:
            best_loss = val_loss
            cur_best_model = model

    loss = cur_best_model.evaluate(ds_test)
    print(best_loss)
    print(loss)

train_optimized_network()