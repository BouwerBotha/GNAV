import Simulator.request
import numpy as np
from common import NUM_SIMS
import tensorflow as tf
NUM_TIMES = 600 #NUM_TIMES*NUM_SIMS = NUMPHOTOS (2000)
k = 0
for i in range(NUM_TIMES):
    pos = np.random.uniform(-0.70,0.70,[NUM_SIMS,1,2])
    angle = np.random.uniform(0.0,360.0,[NUM_SIMS,1])
    target_pos = np.random.uniform(-0.75,0.75,[NUM_SIMS,2])
    avoid_pos = np.random.uniform(-0.75,0.75,[NUM_SIMS,2])
    for i in range(NUM_SIMS):
        while np.linalg.norm(avoid_pos[i]-target_pos[i]) < 0.25:
            avoid_pos[i] = np.random.uniform(-0.75,0.75,[2])

    imgs = Simulator.request.request_images(target_pos,avoid_pos,pos[:,:,0],pos[:,:,1],angle)

    for elem in imgs:
        tf.keras.utils.save_img('images/fake/{0}.png'.format(k),elem[0],scale=False)
        k += 1


Simulator.request.end()