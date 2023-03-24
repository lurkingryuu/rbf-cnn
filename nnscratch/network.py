def predict(network, input):
    '''
    Predicts the output of a network given an input.

    network: a list of layers
    input: the input
    '''
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    '''
    Trains a network on a given dataset using stochastic gradient descent.

    network: a list of layers
    loss: a loss function
    loss_prime: the derivative of the loss function
    x_train: a list of training inputs
    y_train: a list of training outputs
    epochs: the number of training epochs
    learning_rate: the learning rate
    verbose: whether to print the error after each epoch
    '''
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")