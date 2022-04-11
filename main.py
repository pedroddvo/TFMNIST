#!/usr/bin/env python3

import os
import random
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
# the images come as uint8[28][28] arrays of digits 0-9
(train_img, train_lbl), (test_img, test_lbl) = mnist.load_data(path="/media/MNIST/mnist.npz")

# normalize these images for tf
train_img = train_img / 255.0
test_img  = test_img  / 255.0

model = None
if not os.path.isdir(os.getcwd() + "/model"):
    model = tf.keras.Sequential([
        # transform the data from a uint8[28][28] -> uint8[28*28]
        # input layer
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # hidden layer
        tf.keras.layers.Dense(64, activation='relu'),
        # output layer
        tf.keras.layers.Dense(10),
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_img, train_lbl, epochs=5)
    model.save('./model')
else:
    model = tf.keras.models.load_model('./model')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_img)

def plot_test_img(i):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_img[i], cmap=plt.cm.binary)

def plot_bar(i):
    predictions_arr = predictions[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    bar = plt.bar(range(10), predictions_arr, color="#000000")

    bar[np.argmax(predictions_arr)].set_color('red')
    bar[test_lbl[i]].set_color('blue')
    plt.ylim([0, 1])



rows = 5
cols = 5
size = rows*cols

plt.figure(figsize=(2*rows, 2*cols))
for i in range(size):
    # 10,000 test images in the MNIST dataset
    offset = random.randrange(10000)

    plt.subplot(rows, 2*cols, 2*i+1)
    plot_test_img(offset)
    plt.subplot(rows, 2*cols, 2*i+2)
    plot_bar(offset)
plt.tight_layout()
plt.show()
