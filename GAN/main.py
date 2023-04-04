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
        self.noise_dim = 100

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
        model.add(tf.keras.layers.Dense(256, input_dim=self.noise_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.compat.v1.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.compat.v1.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.compat.v1.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(tf.keras.layers.Reshape(self.img_shape))

        model.summary()

        return model

    def build_discriminator(self):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.img_shape))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        return model

    def combined(self, generator, discriminator):

        # generator takes noise as input and generates imgs:https://blog.csdn.net/qq_38669138/article/details/109029782
        discriminator.trainable = False

        noise = tf.keras.layers.Input(shape=(self.noise_dim,))
        img = generator(noise)
        validity = discriminator(img)
        model = tf.keras.Model(noise, validity)

        return model

    def train(self, epochs, batch_size):

        # load the dataset
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

        # rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        # adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            
            if(destory==True): break
            # random sample real image (random.randint(low, high=None, size=None, dtype=int))
            index = np.random.randint(0, x_train.shape[0], batch_size)
            realImage = x_train[index]
            
            # generator generate fake img
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            fakeImage = self.generator(noise)

            # train discriminator --------GAN tip:"How to Train a GAN?" at NIPS2016
            d_loss_real = self.discriminator.train_on_batch(realImage, real)
            d_loss_fake = self.discriminator.train_on_batch(fakeImage, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.generator_discriminator.train_on_batch(noise, real)

            
            text.set("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            # display fake image
            
            if(epoch%100==0):
                self.sample_images(epoch)
                
            window.update()
        
    def sample_images(self, epoch):
        
        row, col = 3, 3
        
        # reset plot
        plt.clf()
        plt.suptitle("Epoch " + str(epoch) + " -- Fake Image", fontsize=28)
        
        # generator generate fake img 
        noise = np.random.normal(0, 1, (row * col, self.noise_dim))
        fakeImage = self.generator(noise)
        
        # rescale images
        fakeImage = 127.5 * (fakeImage + 1)
        
        # plot fakeImage
        cnt = 0
        for i in range(row):
           for j in range(col):
               # The subplot will take the index position on a grid with nrows rows and ncols columns.
               ax = plt.subplot(row, col, cnt+1)
               ax.imshow(fakeImage[cnt, :,:], cmap='gray')
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
window.geometry("750x750")
window.resizable(False, False)
window.title("GAN Algorithm ")
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
fig, ax = plt.subplots(3, 3, figsize = (9, 8), dpi=72)
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