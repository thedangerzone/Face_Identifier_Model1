import numpy as np
import learn


class Layer:
    def __init__(self, activation='linear', input_amount=0, neuron_amount=0):
        self.learn = learn.Learning()
        self.gradientW = np.array([])
        self.gradientB = np.array([])
        self.output = np.array([])
        self.input = np.array([])
        self.inp_to_activation = np.array([])  # z

        self.input_shape = (neuron_amount, input_amount)
        self.weights = np.zeros(self.input_shape)
        self.biases = np.zeros(neuron_amount)

        if activation.lower() == 'relu':
            self.activation_func = self.ReLU
            self.derive_activation = self.Derive_Relu
        elif activation.lower() == 'sigmoid':
            self.activation_func = self.Sigmoid
            self.derive_activation = self.Derive_Sigmoid
        else:
            self.activation_func = self.Linear
            self.derive_activation = self.Derive_Linear

    def Calc_Layer_Output(self, inp):
        self.inp_to_activation = np.dot(self.weights, inp) + self.biases
        self.output = self.activation_func(self.inp_to_activation)
        self.input = np.copy(inp)

        return self.output

    def Calculate_Gradient(self, grad_output, t):
        grad_input = np.dot(grad_output, self.weights.T)
        self.gradientW = np.dot(self.input.T, grad_output)
        if grad_output.shape == ():
            self.gradientB = np.dot(grad_output, self.input.shape[0])
        else:
            self.gradientB = np.dot(grad_output.mean(axis=0), self.input.shape[0])

        self.gradientW = np.resize(self.gradientW, self.weights.shape)
        self.gradientB = np.resize(self.gradientW, self.biases.shape)

        weight_change, bias_change = self.learn.Adam_Optimization(t, self.gradientW, self.gradientB)

        self.weights - weight_change
        self.biases - bias_change

        return grad_input

    def ReLU(self, inp):
        return np.maximum(inp, 0)

    def Sigmoid(self,inp):
        return 1/(1 + pow(np.e,-inp))

    def Linear(self, inp):
        return inp

    def Derive_Relu(self, inp):
        return 1 * (inp > 0)

    def Derive_Sigmoid(self, inp):
        sig = self.Sigmoid(inp)
        return sig - np.dot(sig, sig)

    def Derive_Linear(self, inp):
        return 1



