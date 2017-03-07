"""All the optimization methods go here.

"""

from __future__ import division, print_function, absolute_import
import random
import numpy as np



class SGD(object):
    """Mini-batch stochastic gradient descent.

    Attributes:
        learning_rate(float): the learning rate to use.
        batch_size(int): the number of samples in a mini-batch.

    """

    def __init__(self, learning_rate, batch_size):
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size

    def __has_parameters(self, layer):
        return hasattr(layer, "W")

    def compute_gradient(self, x, y, graph, loss):
        """ Compute the gradients of network parameters (weights and biases)
        using backpropagation.

        Args:
            x(np.array): the input to the network.
            y(np.array): the ground truth of the input.
            graph(obj): the network structure.
            loss(obj): the loss function for the network.

        Returns:
            dv_Ws(list): a list of gradients of the weights.
            dv_bs(list): a list of gradients of the biases.

        """

        # TODO: Backpropagation code
        # print (x.shape)
        inputs=[x]
        for i in graph:
            # print (i.forward(inputs[-1]).shape)
            inputs.append(i.forward(inputs[-1]))

        dv_y=loss.backward(inputs[-1],y)

        dv_Ws = []
        dv_bs = []

        # Traverse List backward, do backpropogation
        for l, i in reversed(zip(graph, inputs[:-1])):
            if type(l).__name__=="FullyConnected":
                dv_y, w, b = l.backward(i, dv_y)
                dv_Ws=[w]+dv_Ws
                dv_bs=[b]+dv_bs
            else:
                dv_y=l.backward(i, dv_y)
        return dv_Ws,dv_bs

    def optimize(self, graph, loss, training_data):
        """ Perform SGD on the network defined by 'graph' using
        'training_data'.

        Args:
            graph(obj): a 'Graph' object that defines the structure of a
                neural network.
            loss(obj): the loss function for the network.
            training_data(list): a list of tuples ``(x, y)`` representing the
                training inputs and the desired outputs.

        """

        # Network parameters
        Ws = [layer.W for layer in graph if self.__has_parameters(layer)]
        bs = [layer.b for layer in graph if self.__has_parameters(layer)]

        # Shuffle the data to make sure samples in each batch are not
        # correlated
        random.shuffle(training_data)
        n = len(training_data)

        batches = [
            training_data[k:k + self.batch_size]
            for k in xrange(0, n, self.batch_size)
        ]

        # TODO: SGD code

        for batch in batches:
            sum_ws=np.zeros(np.array(Ws).shape)
            sum_bs=np.zeros(np.array(bs).shape)
            for row in batch:

                ws, bs = self.compute_gradient(row[0], row[1], graph, loss)
                sum_ws= np.add(sum_ws, ws)
                sum_bs= np.add(sum_bs, bs)

            sum_ws= np.divide(sum_ws, len(batch))
            sum_bs= np.divide(sum_bs, len(batch))
            fullyC=[i for i in graph if type(i).__name__=="FullyConnected"]

            for i in range(len(fullyC)):
                fullyC[i].W = fullyC[i].W-(self.learning_rate*sum_ws[i])
                fullyC[i].b = fullyC[i].b-(self.learning_rate*sum_bs[i])
