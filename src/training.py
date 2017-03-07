import numpy as np
import random
import mnist_loader
from graph import Graph
from loss import Euclidean
from network import Network
from optimization import SGD
import time
from tqdm import tqdm

"""
Quantative Analysis for The DNN

Author : Linxiao Bai

The following functions do different quantitative analysis for different parameter tuning.
Result is analized from factors of precision and convergence speed.
"""

class training:

    @staticmethod
    def tuning_layer(num_layer, size):
        """
        This static method that runs experiment with assigned number of hidden layers. the number of neuron at
         each layer is randomly generated from size-5 to size+5.
        :param num_layer: The number of layer wanted
        :param size: The expected number of width at each layer
        :return: test_precision, training_time(s)
        """
        # Force the first layer to be input of 784
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        graph_config=[]

        input=784
        for i in range(num_layer):
            out=random.randint(size-5,size+5)
            graph_config.append(
                ("FullyConnected", {"shape": (out, input)})
            )
            graph_config.append(
                ("Sigmoid", {})
            )
            input = out

        # Force final to be 10 out puts.
        graph_config.append(
            ("FullyConnected", {"shape": (10, input)})
        )
        graph_config.append(
            ("Sigmoid", {})
        )

        graph = Graph(graph_config)
        loss = Euclidean()
        optimizer = SGD(3.0, 10)  # learning rate 3.0, batch size 10
        network = Network(graph)

        # Train the network for 1 epoch
        start=time.clock()
        network.train(training_data, 1, loss, optimizer)
        elipse=time.clock()-start


        correct=0.
        for x,y in test_data:
            if y== np.argmax(network.inference(x)):
                correct+=1
        return correct/len(test_data), elipse


if __name__ == '__main__':

    f1 = open('/Users/Pagliacci/Desktop/tuning_layer_precision.csv', 'w')
    f2 = open('/Users/Pagliacci/Desktop/tuning_layer_time.csv', 'w')
    for layerN in tqdm(range (1,11)):
        for sizeL in tqdm(range (10,110, 10)):
            prec, t=training.tuning_layer(layerN, sizeL)
            f1.write(str(layerN)+"," + str(sizeL) + "," + str(prec)+"\n")
            f2.write(str(layerN)+"," + str(sizeL) + "," + str(t)+"\n")
    f1.close()
    f2.close()






