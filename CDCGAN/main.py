import sys
import numpy as np
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt

from tkinter import ttk 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.python.client import device_lib

# https://blog.csdn.net/Strive_For_Future/article/details/115243512
# from tensorflow.compat.v1.keras.layers import BatchNormalization

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.noise_dim = 100
        self.embedding_dim = 200

        optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)

        # build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # build generator
        self.generator = self.build_generator()

        # build generator_discriminator(stacked generator and discriminator)
        self.generator_discriminator = self.combined(self.generator, self.discriminator)
        self.generator_discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(128 * 7 * 7, activation="relu"))
        model.add(tf.keras.layers.Reshape((7, 7, 128)))
        model.add(tf.keras.layers.UpSampling2D())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
        model.add(tf.compat.v1.keras.layers.BatchNormalization(momentum=0.001))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.UpSampling2D())
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
        model.add(tf.compat.v1.keras.layers.BatchNormalization(momentum=0.001))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.Activation("tanh"))
        
        noise = tf.keras.layers.Input(shape=(self.noise_dim,))
        label = tf.keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(self.num_classes, self.embedding_dim)(label))
        model_input = tf.keras.layers.Concatenate()([noise, label_embedding])
        fakeImage = model(model_input)
        
        model = tf.keras.Model([noise, label], fakeImage)
        model.summary()
        
        return model

    def build_discriminator(self):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(tf.compat.v1.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(tf.compat.v1.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(tf.compat.v1.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        
        fakeImage = tf.keras.layers.Input(shape=self.img_shape)
        flat_fakeImage = model(fakeImage)
        
        label = tf.keras.layers.Input(shape=(1,), dtype='int32')
        label_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(self.num_classes, self.embedding_dim)(label))
        
        x = tf.keras.layers.Concatenate()([flat_fakeImage, label_embedding])
        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        validity = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model([fakeImage, label], validity)
        model.summary()
        
        return model

    def combined(self, generator, discriminator):

        # generator takes noise as input and generates imgs
        discriminator.trainable = False

        noise = tf.keras.layers.Input(shape=(self.noise_dim,))
        label = tf.keras.layers.Input(shape=(1,), dtype='int32')
        fakeImage = generator([noise, label])
        validity = discriminator([fakeImage, label])
        
        model = tf.keras.Model([noise, label], validity)

        return model

    def train(self, epochs, batch_size):

        # load the dataset
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

        # rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        y_train = y_train.reshape(-1, 1)
        
        # adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            
            if(destory==True): break
            # random sample real image (random.randint(low, high=None, size=None, dtype=int))
            index = np.random.randint(0, x_train.shape[0], batch_size)
            realImage, label = x_train[index], y_train[index]
            
            # generator generate fake img
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fakeImage = self.generator.predict([noise, label])

            # train discriminator --------GAN tip:"How to Train a GAN?" at NIPS2016
            d_loss_real = self.discriminator.train_on_batch([realImage, label], real)
            d_loss_fake = self.discriminator.train_on_batch([fakeImage, label], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            sampled_label = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            g_loss = self.generator_discriminator.train_on_batch([noise, sampled_label], real)

            text.set("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            # display fake image
            if(epoch%100==0):
                self.sample_images(epoch)
                
            window.update()
        
    def sample_images(self, epoch):
        
        row, col = 2, 5
    
        # reset plot
        plt.clf()
        plt.suptitle("Epoch " + str(epoch) + " -- Fake Image", fontsize=24)
        
        # generator generate fake img 
        noise = np.random.normal(0, 1, (row * col, self.noise_dim))
        sampled_label = np.arange(0, 10).reshape(-1, 1)
        fakeImage = self.generator.predict([noise, sampled_label])
        
        # rescale images 0~1, for imshow with RGB data ([0..1] for floats or [0..255] for integers). 
        fakeImage = 0.5 * fakeImage + 0.5
        
        # plot fakeImage
        cnt = 0
        for _ in range(row):
           for _ in range(col):
               # The subplot will take the index position on a grid with nrows rows and ncols columns.
               ax = plt.subplot(row, col, cnt+1)
               ax.imshow(fakeImage[cnt, :,:], cmap='gray')
               ax.set_title(str(cnt), fontsize=14)
               ax.axis('off')
               cnt += 1

        canvas.draw()

# disable Buttom & Entry
def disable(component):
    component['state'] = 'disable'

print("Environment confirmation:", sys.executable)
print("TensorFlow version:", tf.__version__)
print("-------------------------------------------------------------------------------------------------------------------")
print("Check devices available to tensorFlow")
print(device_lib.list_local_devices())
print("-------------------------------------------------------------------------------------------------------------------")
print("Hardware devices")
print(tf.config.list_physical_devices(device_type='CPU'))
print(tf.config.list_physical_devices(device_type='GPU'))
print("-------------------------------------------------------------------------------------------------------------------")
print("Usable CPU or GPU (visible_devices) by tensorflow")
print(tf.config.get_visible_devices())
print("-------------------------------------------------------------------------------------------------------------------")

window = tk.Tk()
window.geometry("750x530")
window.resizable(False, False)
window.title("Conditional Convolution GAN")
window.configure(bg='#E6E6FA')

# Global var
destory = False
epoch = tk.IntVar()
epoch.set(30000)
batch_size = tk.IntVar()
batch_size.set(32)
text = tk.StringVar()
text.set("-- [D loss: --, acc.: --%] [G loss: --]")

# tk Frame
frame1 = tk.Frame(window, bg="#F0FFF0")
frame1.pack(side='top', pady=10)
separator = ttk.Separator(window, orient='horizontal')
separator.pack(side='top', fill=tk.X)
frame2 = tk.Frame(window)
frame2.pack(side='top', pady=10)
frame3 = tk.Frame(window)
frame3.pack(side='top', pady=10)

# Plot
fig, ax = plt.subplots(2, 5, figsize = (9, 5), dpi=72)
canvas = FigureCanvasTkAgg(fig, frame3)  # A tk.DrawingArea.
canvas.get_tk_widget().grid()

# Algorithm
gan = GAN()
gan.sample_images(0)

# GUI
tk.Label(frame1, font=("Calibri", 15, "bold"), text="Epochs:", bg="#F0FFF0").pack(side='left', padx=5)
ent1 = tk.Entry(frame1, width=8, textvariable=epoch)
ent1.pack(side='left')

tk.Label(frame1, font=("Calibri", 15, "bold"), text="Batch Size:", bg="#F0FFF0").pack(side='left', padx=5)
ent2 = tk.Entry(frame1, width=5, textvariable=batch_size)
ent2.pack(side='left')

click = 0
btn1 = tk.Button(frame1, font=("Calibri", 12, "bold"), text='GAN Train', command=lambda:[disable(btn1), disable(ent1), disable(ent2), gan.train(epoch.get(), batch_size.get())])
btn1.pack(side='left', padx=(10, 5), pady=5)

label1 = tk.Label(frame2, font=("Calibri", 15, "bold"), textvariable=text, bg="#F0FFF0")
label1.pack()

# repair window close bug
def destory_tk():
    global destory
    destory = True    
    window.destroy()

window.protocol("WM_DELETE_WINDOW", destory_tk)
window.mainloop()