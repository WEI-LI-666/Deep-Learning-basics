#/usr/bin/python3
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

dense_layers = [0, 1, 2]
layer_sizes = [32, 64]
conv_layers= [1, 2, 3]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess= tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='../logs/{}'.format(NAME))
            #print(NAME)
            model = Sequential()
            model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size = (2, 2)))

            for i in range(conv_layer-1):
                model.add(Conv2D(64, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size = (2, 2)))

            model.add(Flatten())

            for i in range(dense_layer-1):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])
            model.fit(X, y, batch_size=32, epochs=15, validation_split=0.1, callbacks=[tensorboard])