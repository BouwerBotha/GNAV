import tensorflow as tf
import data
import network
import numpy as np
snn = network.get_model()

snn.load_weights('checkpoints/snn.ckpt')

ds_train, ds_val, ds_test = data.get_data()

snn.evaluate(ds_test)

output  = snn.predict(np.array([[1,1]]))
print(output)