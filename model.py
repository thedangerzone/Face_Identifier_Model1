import numpy as np
import pickle
import Layer
import learn
import os
from time import time


class Model:
    def __init__(self):
        self.learn = learn.Learning()
        self.layers = np.array([])
        self.loss = 0
        self.accuracy = 0
        self.default_path = os.environ.get('userprofile')+r'\Documents\Model'

    def train(self, input_train, label_train, input_test, label_test, epochs=300, print_every=10, model_save_location=r'C:\Users\omers\Documents\Model'):
        for epoch in range(epochs):  #go over all epochs
            start = time()

            # shuffle the dataset input
            sample_amount = len(input_train)
            shuffler = np.random.permutation(sample_amount)
            print('input_train', input_train)
            shuffled_input = input_train[shuffler]
            print('label_train', label_train)
            shuffled_labels = label_train[shuffler]

            for sample in range(sample_amount):  # go over all samples of data in dataset
                model_output = shuffled_input[sample]
                for layer_index in range(len(self.layers)):
                    model_output = self.layers[layer_index].Calc_Layer_Output(model_output)
                self.loss = self.learn.Binary_Cross_Entropy(labels=shuffled_labels, preds=model_output,
                                                            step=1, starting_sample=sample)
                loss_gradient = self.learn.Derived_Binary_Cross_Entropy(labels=shuffled_labels[sample], preds=model_output,
                                                                        step=1, starting_sample=sample)
                amount_of_samples_done = (sample+1) + epoch*sample_amount
                for layer_index in range(len(self.layers)-1, 0, -1):
                    loss_gradient = self.layers[layer_index].Calculate_Gradient(loss_gradient, t=amount_of_samples_done)

                #amount_of_samples_done = (sample+1) + epoch*sample_amount
                #self.learn.Adam_Optimization(t=amount_of_samples_done, layers=self.layers)  # layer1 doesn't have weights to update

            end = time()
            if (epoch+1) % print_every == 0 or True:
                sample_test_amount = len(input_train)
                shuffler2 = np.random.permutation(sample_test_amount)
                input_test = input_test[shuffler2]
                label_test = label_test[shuffler2]

                avgloss = 0
                correct_amount = 0

                for sample in range(sample_test_amount):  # go over all samples of data in dataset
                    model_output = input_test[sample]
                    for layer_index in range(1, len(self.layers)):
                        model_output = self.layers[layer_index].Calc_Layer_Output(model_output)
                    avgloss += self.learn.Binary_Cross_Entropy(labels=label_test[sample], preds=model_output, sample=1)
                    if round(model_output) == label_test[sample]:
                        correct_amount += 1

                self.accuracy = correct_amount/sample_test_amount
                avgloss /= sample_test_amount

                print(f'epoch:{epoch},loss{avgloss},'
                      f'accuracy:{self.accuracy},'
                      f'epoch time{round(end-start,2)}')
                if not os.path.exists(model_save_location):
                    os.mkdir(model_save_location)
                self.save(f'{model_save_location}\\+Model{e}')

        if not os.path.exists(model_save_location):
            os.mkdir(model_save_location)
        self.save(f'{model_save_location}\\+Model')

    def Add_Layer(self, layer):
        self.layers = np.append(self.layers, layer)

    def save(self, path):
        weights = []
        biases = []
        for layer in self.layers:
            weights.append(layer.weights)
            biases.append(layer.biass)

        params = {'w': weights, 'b': biases}
        pickle.dump(params, open(path + '.p', 'wb'))

    def load(self, path):
        params = pickle.load(open(path, 'rb'))
        weights = params['w']
        biases = params['b']

        for i, weights_and_biases in enumerate(zip(weights, biases)):
            self.layers[i].setW(weights_and_biases[0])
            self.layers[i].setb(weights_and_biases[1])

