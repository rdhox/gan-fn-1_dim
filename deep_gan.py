#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:51:00 2021

@author: renaud
"""

import numpy as np
from matplotlib import pyplot
from numpy.random import rand, randn
from numpy import hstack, ones, zeros

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle 

def calculate(x):
    return x*x

# generate randoms sample from x^2
def generate_real_samples(n=100):
	# generate random inputs in [-0.5, 0.5]
    # X1 = rand(n) - 0.5
    a, b = -4, 4
    X1 = a + rand(n)*(b-a)
	# generate outputs X^2 (quadratic)
    X2 = np.cos(X1)
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = ones((n, 1))
    return X, y

def generate_fake_samples_simple(n=100):
 	# generate random inputs in [-1, 1]
    X1 = - 1 + rand(n)*2
    X2 = - 1 + rand(n)*2
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = zeros((n, 1))
    return X, y

# Discriminator model
def define_discriminator(n_input=2):
    model = Sequential()
    model.add(Dense(150, activation='relu', kernel_initializer='he_uniform', input_dim=n_input))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model

# generate point in the latent space for generator
def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim*n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    # plot the results
    # pyplot.scatter(X[:, 0], X[:, 1])
    # pyplot.show()
    y = zeros((n, 1))
    return X, y

# define the combined generator and discriminator model, updating the generator
def define_gan(generator, discriminator):
    # make weights for the discriminator not trainable
    discriminator.trainable = False
    # create new gan model
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# Train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
    half_batch = int(n_batch/2)
    summarize_performance(0, g_model, d_model, latent_dim)
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator (weight trainable on train_on_batch)
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        gan_model.train_on_batch(x_gan, y_gan)
        if (i+1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)

# plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    x_real, y_real = generate_real_samples(n)
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print(epoch, acc_real, acc_fake)
    # scatter plot real and fake data points
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    pyplot.title(f'epoch: {epoch}')
    pyplot.show()

# plot the discriminator
# plot_model(discriminator, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
discriminator = define_discriminator()

# define the generator model
latent_dim = 5
generator = define_generator(latent_dim)
# summarize the model
# plot the generator
# plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# define gan model
gan = define_gan(generator, discriminator)
# plot_model(generator, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

train(generator, discriminator, gan, latent_dim)




    