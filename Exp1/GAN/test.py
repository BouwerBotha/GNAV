from simple_network import get_generator,get_discriminator,CycleGan,generator_loss_fn,discriminator_loss_fn
import keras
import Simulator.request
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from common import NUM_SIMS,POPULATION_SIZE,HEIGHT,WIDTH,NUM_DIM
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

cycle_gan_model.load_weights('model_checkpoints/3/cyclegan_checkpoints.100')
generator = cycle_gan_model.gen_G

cycle_gan_model = None
pos = np.random.uniform(-0.70, 0.70, [NUM_SIMS, 1, 2])
angle = np.random.uniform(0.0, 360.0, [NUM_SIMS, 1])
target_pos = np.random.uniform(-0.75, 0.75, [NUM_SIMS, 2])
avoid_pos = np.random.uniform(-0.75, 0.75, [NUM_SIMS, 2])


imgs = Simulator.request.request_images(target_pos, avoid_pos, pos[:, :, 0], pos[:, :, 1], angle)

imgs_real = (imgs/127.5 - 1.0).astype(np.float32)

imgs_real = tf.reshape(imgs_real,[NUM_SIMS*1,HEIGHT,WIDTH,NUM_DIM])

imgs_real = gen_G(imgs_real)

imgs_real = tf.reshape(imgs_real,[NUM_SIMS,1,HEIGHT,WIDTH,NUM_DIM])

imgs_real = (imgs_real * 127.5 + 127.5).numpy().astype(np.uint8)

_, ax = plt.subplots(NUM_SIMS, 2, figsize=(12, 12))

for i in range(NUM_SIMS):

    prediction = imgs_real[i,0]
    img = imgs[i,0]

    ax[i, 0].imshow(img)
    ax[i, 1].imshow(prediction)
    ax[i, 0].set_title("Input image")
    ax[i, 1].set_title("Translated image")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")

plt.show()

Simulator.request.end()