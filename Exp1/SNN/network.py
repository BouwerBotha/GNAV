import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import LeakyReLU

###[+-5000,15000]

def get_activation(string):
    if string == "relu":
        return keras.layers.ReLU()
    elif string == "leaky":
        return keras.layers.LeakyReLU()
    else:
        return keras.layers.PReLU()
##used for hyperparameter tuning
def build_model(hp):

    model = keras.Sequential()

    model.add(layers.Input(shape=(2,)))
    num_units = hp.Int('unitsperlayer',min_value=20, max_value=100,step=20)
    act = hp.Choice('activation',values=["relu","leaky","prelu"])
    # dropout_perc = hp.Choice('dropout_perc', values=[0.0, 0.1, 0.2])
    for i in range(1,hp.Int('num_layers', min_value=2, max_value = 4, step=1)):
        model.add(layers.Dense(num_units,activation=get_activation(act)))
        # if dropout_perc != 0.0:
        #     model.add(layers.Dropout(rate=dropout_perc))
    model.add(layers.Dense(3, activation='tanh'))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values = [0.01, 0.005, 0.001])), loss=tf.keras.losses.MeanSquaredError())
    return model
##optimized model
def get_model():

    model = keras.Sequential()
    model.add(layers.Input(shape=(2,)))
    model.add(layers.Dense(80,activation=layers.PReLU()))
    model.add(layers.Dense(80,activation=layers.PReLU()))
    model.add(layers.Dense(3, activation='tanh'))
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.MeanSquaredError())
#
#     return model

    return model


# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 50
# hlayer1act: prelu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.0005
# hlayer2units: 80
# hlayer2act: prelu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 100
# hlayer3act: relu
# dropout_perc_hlayer3: 0.2
# hlayer4units: 90
# hlayer4act: relu
# dropout_perc_hlayer4: 0.0
# hlayer5units: 60
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 20
# hlayer6act: relu
# dropout_perc_hlayer6: 0.2
# hlayer7units: 60
# hlayer7act: leaky
# dropout_perc_hlayer7: 0.0
# tuner/epochs: 1000
# tuner/initial_epoch: 334
# tuner/bracket: 5
# tuner/round: 5
# tuner/trial_id: 1733
# Score: 0.0002735970076173544
# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 50
# hlayer1act: prelu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.0005
# hlayer2units: 80
# hlayer2act: prelu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 100
# hlayer3act: relu
# dropout_perc_hlayer3: 0.2
# hlayer4units: 90
# hlayer4act: relu
# dropout_perc_hlayer4: 0.0
# hlayer5units: 60
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 20
# hlayer6act: relu
# dropout_perc_hlayer6: 0.2
# hlayer7units: 60
# hlayer7act: leaky
# dropout_perc_hlayer7: 0.0
# tuner/epochs: 112
# tuner/initial_epoch: 38
# tuner/bracket: 5
# tuner/round: 3
# tuner/trial_id: 1689
# Score: 0.00027578804292716086
# Trial summary
# Hyperparameters:
# num_layers: 7
# hlayer1units: 50
# hlayer1act: leaky
# dropout_perc_hlayer1: 0.1
# learning_rate: 0.0005
# hlayer2units: 70
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 30
# hlayer3act: leaky
# dropout_perc_hlayer3: 0.0
# hlayer4units: 40
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.1
# hlayer5units: 50
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 100
# hlayer6act: prelu
# dropout_perc_hlayer6: 0.1
# hlayer7units: 40
# hlayer7act: leaky
# dropout_perc_hlayer7: 0.2
# tuner/epochs: 112
# tuner/initial_epoch: 38
# tuner/bracket: 6
# tuner/round: 4
# tuner/trial_id: 1217
# Score: 0.0002806421834975481
# Trial summary
# Hyperparameters:
# num_layers: 3
# hlayer1units: 100
# hlayer1act: relu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.001
# hlayer2units: 90
# hlayer2act: prelu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 40
# hlayer3act: leaky
# dropout_perc_hlayer3: 0.0
# hlayer4units: 30
# hlayer4act: relu
# dropout_perc_hlayer4: 0.1
# hlayer5units: 40
# hlayer5act: relu
# dropout_perc_hlayer5: 0.1
# hlayer6units: 30
# hlayer6act: leaky
# dropout_perc_hlayer6: 0.1
# hlayer7units: 40
# hlayer7act: relu
# dropout_perc_hlayer7: 0.0
# tuner/epochs: 1000
# tuner/initial_epoch: 334
# tuner/bracket: 5
# tuner/round: 5
# tuner/trial_id: 1734
# Score: 0.00028247853915672747
# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 90
# hlayer1act: prelu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.001
# hlayer2units: 70
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 70
# hlayer3act: prelu
# dropout_perc_hlayer3: 0.0
# hlayer4units: 40
# hlayer4act: prelu
# dropout_perc_hlayer4: 0.0
# hlayer5units: 40
# hlayer5act: relu
# dropout_perc_hlayer5: 0.2
# hlayer6units: 70
# hlayer6act: prelu
# dropout_perc_hlayer6: 0.0
# hlayer7units: 60
# hlayer7act: relu
# dropout_perc_hlayer7: 0.0
# tuner/epochs: 38
# tuner/initial_epoch: 0
# tuner/bracket: 3
# tuner/round: 0
# Score: 0.0002835148188751191
# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 40
# hlayer1act: relu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.005
# hlayer2units: 60
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 60
# hlayer3act: relu
# dropout_perc_hlayer3: 0.1
# hlayer4units: 40
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.2
# hlayer5units: 20
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 50
# hlayer6act: relu
# dropout_perc_hlayer6: 0.1
# hlayer7units: 70
# hlayer7act: prelu
# dropout_perc_hlayer7: 0.2
# tuner/epochs: 38
# tuner/initial_epoch: 0
# tuner/bracket: 3
# tuner/round: 0
# Score: 0.0002836827770806849
# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 40
# hlayer1act: relu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.005
# hlayer2units: 60
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 60
# hlayer3act: relu
# dropout_perc_hlayer3: 0.1
# hlayer4units: 40
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.2
# hlayer5units: 20
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 50
# hlayer6act: relu
# dropout_perc_hlayer6: 0.1
# hlayer7units: 70
# hlayer7act: prelu
# dropout_perc_hlayer7: 0.2
# tuner/epochs: 112
# tuner/initial_epoch: 38
# tuner/bracket: 3
# tuner/round: 1
# tuner/trial_id: 1937
# Score: 0.0002839843276888132
# Trial summary
# Hyperparameters:
# num_layers: 8
# hlayer1units: 90
# hlayer1act: leaky
# dropout_perc_hlayer1: 0.2
# learning_rate: 0.005
# hlayer2units: 70
# hlayer2act: relu
# dropout_perc_hlayer2: 0.1
# hlayer3units: 60
# hlayer3act: relu
# dropout_perc_hlayer3: 0.0
# hlayer4units: 70
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.1
# hlayer5units: 100
# hlayer5act: leaky
# dropout_perc_hlayer5: 0.2
# hlayer6units: 90
# hlayer6act: relu
# dropout_perc_hlayer6: 0.0
# hlayer7units: 60
# hlayer7act: leaky
# dropout_perc_hlayer7: 0.1
# tuner/epochs: 112
# tuner/initial_epoch: 38
# tuner/bracket: 3
# tuner/round: 1
# tuner/trial_id: 1960
# Score: 0.00028502397472038863
# Trial summary
# Hyperparameters:
# num_layers: 6
# hlayer1units: 50
# hlayer1act: prelu
# dropout_perc_hlayer1: 0.1
# learning_rate: 0.005
# hlayer2units: 70
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 70
# hlayer3act: prelu
# dropout_perc_hlayer3: 0.0
# hlayer4units: 30
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.0
# hlayer5units: 60
# hlayer5act: relu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 80
# hlayer6act: prelu
# dropout_perc_hlayer6: 0.2
# hlayer7units: 20
# hlayer7act: relu
# dropout_perc_hlayer7: 0.0
# tuner/epochs: 1000
# tuner/initial_epoch: 334
# tuner/bracket: 2
# tuner/round: 2
# tuner/trial_id: 2045
# Score: 0.0002868462528567761
# Trial summary
# Hyperparameters:
# num_layers: 4
# hlayer1units: 40
# hlayer1act: relu
# dropout_perc_hlayer1: 0.0
# learning_rate: 0.005
# hlayer2units: 60
# hlayer2act: relu
# dropout_perc_hlayer2: 0.0
# hlayer3units: 60
# hlayer3act: relu
# dropout_perc_hlayer3: 0.1
# hlayer4units: 40
# hlayer4act: leaky
# dropout_perc_hlayer4: 0.2
# hlayer5units: 20
# hlayer5act: prelu
# dropout_perc_hlayer5: 0.0
# hlayer6units: 50
# hlayer6act: relu
# dropout_perc_hlayer6: 0.1
# hlayer7units: 70
# hlayer7act: prelu
# dropout_perc_hlayer7: 0.2
# tuner/epochs: 334
# tuner/initial_epoch: 112
# tuner/bracket: 3
# tuner/round: 2
# tuner/trial_id: 1990
# Score: 0.0002878060913644731