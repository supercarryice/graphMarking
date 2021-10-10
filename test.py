import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
import keras.backend as K

num_classes = 1
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#print(x_train.shape)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape: ",x_train.shape)
print("x_test shape: ",x_test.shape)
#print(y_train[0])
#print("old y_train shape", y_train.shape)
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#print("new y_train shape", y_train.shape)
#print(y_train[0])

# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32,kernel_size=(3,3),activation="relu"),
#         layers.MaxPool2D(pool_size=(2,2)),
#         layers.Conv2D(64,kernel_size=(3,3),activation="relu"),
#         layers.MaxPool2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes)
#     ]
# )
batch_size = 128
epochs = 5


inp = layers.Input(shape=input_shape)
x = layers.Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation="relu",
)(inp)

x = layers.MaxPool2D(
    pool_size=(2,2),
)(x)

x = layers.Conv2D(
    filters=64,
    kernel_size=(3,3),
    activation="relu",
)(x)

x = layers.MaxPool2D(
    pool_size=(2,2),
)(x)

x = layers.Flatten()(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(num_classes)(x)

def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


tfboard = TensorBoard(log_dir="test_model", histogram_freq=0, write_grads=True)
model = keras.models.Model(inp, x)
model.compile(loss="mse", optimizer="adam", metrics=["mae", r2])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tfboard])

save_path = './saved_weights'
model.save_weights(save_path)

# load_model = keras.models.Model(inp, x)
# load_model.load_weights(save_path)
#
# print("model load success")
# pred_test_y = load_model.predict(x_test)
# print(pred_test_y[0:10])
# print(y_test[0:10])