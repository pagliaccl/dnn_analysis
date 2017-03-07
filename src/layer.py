"""All the layer functions go here.
"""

from __future__ import division, print_function, absolute_import
import numpy as np


class FullyConnected(object):
    """Fully connected layer 'y = Wx + b'.

    Arguments:
        shape(tuple): the shape of the fully connected layer. shape[0] is the
            output size and shape[1] is the input size.

    Attributes:
        W(np.array): the weights of the fully connected layer. An n-by-m matrix
            where m is the input size and n is the output size.
        b(np.array): the biases of the fully connected layer. A n-by-1 vector
            where n is the output size.

    """

    def __init__(self, shape):
        self.W = np.random.randn(*shape)
        self.b = np.random.randn(shape[0], 1)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """

        return np.dot(self.W, x)+self.b

    def backward(self, x, dv_y):
        """Compute the gradients of weights and biases and the gradient with
        respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            dv_x(np.array): The derivative of the loss with respect to the
                input.
            dv_W(np.array): The derivative of the loss with respect to the
                weights.
            dv_b(np.array): The derivative of the loss with respect to the
                biases.

        """
        return (np.dot(self.W.T, dv_y), np.dot(dv_y, np.transpose(x)), dv_y)


class Sigmoid(object):
    """Sigmoid function 'y = 1 / (1 + exp(-x))'

    """

    def forward(self, x):
        """Compute the layer output.

        Args:
            x(np.array): the input of the layer.

        Returns:
            The output of the layer.

        """
        return 1 / (1 + np.exp(-x))


    def backward(self, x, dv_y):
        """Compute the gradient with respect to the input.

        Args:
            x(np.array): the input of the layer.
            dv_y(np.array): The derivative of the loss with respect to the
                output.

        Returns:
            The derivative of the loss with respect to the input.

        """

        dy_x=(np.exp(-x))/((1+np.exp(-x)) ** 2)
        return dv_y * dy_x