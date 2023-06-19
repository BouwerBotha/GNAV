import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import layers
import GAN.data as data
import time


# # ds_fake, ds_real = data.get_data()
# test_real = []
# test_fake = []
# model = 1
# for file in os.listdir('images/test_real'):
#     name = 'images/test_real/' + file
#     im = Image.open(name)
#     im = np.array(im)
#     im = im/127.5 - 1.0
#     test_real.append(im)
#
# for file in os.listdir('images/test_fake'):
#     name = 'images/test_fake/' + file
#     im = Image.open(name)
#     im = np.array(im)
#     im = im / 127.5 - 1.0
#     test_fake.append(im)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def residual_block(
    x,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer = kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)
    x = activation(x)

    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer = kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
    x,
    filters,
    activation,
    kernel_initializer=kernel_init,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,

        padding=padding,
        use_bias=use_bias,kernel_initializer=kernel_initializer,
    )(x)
    if activation:
        x = activation(x)
    return x


def upsample(
    x,
    filters,
    activation,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="same",
    kernel_initializer=kernel_init,
    gamma_initializer=gamma_init,
    use_bias=False,
):
    x = layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias,
    )(x)
    if activation:
        x = activation(x)
    return x

def get_generator(
    filters=64,
    num_downsampling_blocks=3,
    num_residual_blocks=1,
    num_upsample_blocks=3,
    gamma_initializer=gamma_init,
    name=None,
):
    img_input = layers.Input(shape=(48,64,3), name=name + "_img_input")
    x = layers.Conv2D(filters/2,kernel_size=(7,7),padding="same", kernel_initializer=kernel_init)(img_input)
    x = layers.LeakyReLU()(x)
    # Downsampling
    # skips=[]
    for _ in range(num_downsampling_blocks):

        x = downsample(x, filters=filters, activation=layers.LeakyReLU())
        # skips.append(x)
        filters *= 2


    # skips.reverse()

    # Residual blocks
    for i in range(num_residual_blocks):
        x = residual_block(x, activation=layers.LeakyReLU())

    filters//=2
    # Upsampling
    for i in range(num_upsample_blocks):
        filters //= 2
        # x = layers.concatenate([x, skips[i]])
        x = upsample(x, filters, activation=layers.LeakyReLU())


    # Final block
    x = layers.Conv2D(3, (7, 7), padding="same", kernel_initializer=kernel_init)(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model

def get_discriminator(
    filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None
):
    img_input = layers.Input(shape=(48,64,3), name=name + "_img_input")
    x = img_input

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(),
                kernel_size=(4, 4),
                strides=(2, 2),
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=layers.LeakyReLU(),
                kernel_size=(4, 4),
                strides=(1, 1),
            )

    x = layers.Conv2D(
        1, (7, 7), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model



# Get the generators
# gen_G = get_generator(name="generator_G")
# print(gen_G.summary())
# gen_F = get_generator(name="generator_F")
#
# # Get the discriminators
# disc_X = get_discriminator(name="discriminator_X")
# print(disc_X.summary())
# disc_Y = get_discriminator(name="discriminator_Y")

class CycleGan(keras.Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        discriminator_X,
        discriminator_Y,
        lambda_cycle=10.0,
        lambda_identity=5.0,
    ):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
        disc_X_optimizer,
        disc_Y_optimizer,
        gen_loss_fn,
        disc_loss_fn,
    ):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        real_x, real_y = batch_data



        with tf.GradientTape(persistent=True) as tape:

            fake_y = self.gen_G(real_x, training=True)

            fake_x = self.gen_F(real_y, training=True)


            cycled_x = self.gen_F(fake_y, training=True)

            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (self.identity_loss_fn(real_y, same_y)* self.lambda_identity)
            id_loss_F = (self.identity_loss_fn(real_x, same_x)* self.lambda_identity)

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            _, ax = plt.subplots(10, 2, figsize=(12, 12))
            for i, img in enumerate(test_fake):
                prediction = self.model.gen_G(np.expand_dims(img,axis=0))[0]

                prediction = (prediction * 127.5 + 127.5).numpy().astype(np.uint8)
                img = (img * 127.5 + 127.5).astype(np.uint8)

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")



            plt.savefig('progress/{0}/genX_{1}.png'.format(model,epoch))
            plt.close()
            _, ax = plt.subplots(10, 2, figsize=(12, 12))
            for i, img in enumerate(test_real):
                prediction = self.model.gen_F(np.expand_dims(img, axis=0))[0]
                prediction = (prediction * 127.5 + 127.5).numpy().astype(np.uint8)
                img = (img * 127.5 + 127.5).astype(np.uint8)

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")



            plt.savefig('progress/{0}/genY_{1}.png'.format(model,epoch))
            plt.close()

# adv_loss_fn = keras.losses.MeanSquaredError()
adv_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5



if __name__ == "__main__":
    while True:
        gen_G = get_generator(name="generator_G")
        print(gen_G.summary())
        gen_F = get_generator(name="generator_G")


        disc_X = get_discriminator(name="discriminator_X")
        print(disc_X.summary())
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
        # Callbacks
        plotter = GANMonitor()
        checkpoint_filepath = "./model_checkpoints/{0}".format(model)+"/cyclegan_checkpoints.{epoch:03d}"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True,save_freq = 374*5
        )

    # Here we will train the model for just one epoch as each epoch takes around
    # 7 minutes on a single P100 backed machine.

        cycle_gan_model.fit(
            tf.data.Dataset.zip((ds_fake, ds_real)),
            epochs=100,
            callbacks=[plotter,model_checkpoint_callback],
        )
        model+=1
        if model == 6:
            break




