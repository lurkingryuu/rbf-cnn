import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from nnscratch.dense import Dense
from nnscratch.convolutional import Convolutional
from nnscratch.maxpool import Maxpooling
from nnscratch.reshape import Reshape
from nnscratch.activations import ReLu, Softmax
from nnscratch.losses import binary_cross_entropy, binary_cross_entropy_prime
from nnscratch.network import train, predict

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 16),
    Maxpooling(),
    ReLu(),
    Convolutional((1, 13, 13), 3, 16),
    Maxpooling(),
    ReLu(),
    Convolutional((1, 5, 5), 3, 8),
    Maxpooling(),
    ReLu(),
    Reshape((256, 5, 5), (256 * 5 * 5, 1)),
    Dense(256 * 5 * 5, 10),
    ReLu(),
    Dense(10, 10),
    Softmax(),
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")