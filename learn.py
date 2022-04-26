import numpy as np
import Layer


class Learning:
    def __init__(self, beta1=0.9, beta2=0.9999, epsilon=5e-4, learning_rate=5e-4, amount_of_layers=0):
        empty = np.array([0] * amount_of_layers) #empty momentum and RMSProp array pattern
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.loss = 0
        self.momentum_w, self.RMSProp_w = np.copy(empty), np.copy(empty)
        self.momentum_b, self.RMSProp_b = np.copy(empty), np.copy(empty)

    def Binary_Cross_Entropy(self, labels, preds, step, starting_sample):
        for sample in range(step):
            current = sample + starting_sample
            if labels[current] == 1:
                self.loss += np.log(preds[current])
            else:
                self.loss += np.log(1-preds[current])
        self.loss *= -1/step

        return self.loss

    def Derived_Binary_Cross_Entropy(self, labels, preds, step, starting_sample):
        dl_to_do = 0  # difference in loss to output
        for sample in range(step):
            current = sample + starting_sample
            if labels[current] == 1:
                dl_to_do += 1 / preds[current]
            else:
                dl_to_do += -1 / (1-preds[current])
        dl_to_do *= -1/step

        return dl_to_do

    def Adam_Optimization(self, t, layers):
        for i, layer in enumerate(layers):
            dw = layer.gradientW
            db = layer.gradientB
            # momentum - beta1
            self.momentum_w[i] = np.dot(self.beta1, self.momentum_w[i]) + np.dot((1-self.beta1), dw)  # for weights
            self.momentum_b[i] = np.dot(self.beta1, self.momentum_b[i]) + np.dot((1-self.beta1), db)  # for biases

            # RMSProp - beta2
            self.RMSProp_w[i] = np.dot(self.beta2 * self.RMSProp_w[i]) + np.dot((1 - self.beta2), (dw**2))  # for weights
            self.RMSProp_b[i] = np.dot(self.beta2 * self.RMSProp_b[i]) + np.dot((1 - self.beta2), (db**2))  # for biases

            # bias correction
            moment_w = np.dot(self.momentum_w[i], 1 / (1-self.beta1**t))  # correct momentum for weights
            moment_b = np.dot(self.momentum_b[i], 1 / (1-self.beta1**t))  # correct momentum for biases
            RMS_w = np.dot(self.RMS_w[i], 1 / (1-self.beta2**t))  # correct RMSProp for weights
            RMS_b = np.dot(self.RMS_b[i], 1 / (1-self.beta2**t))  # correct RMSProp for biases

            # update params
            layer.weights = layer.weights - np.dot(self.learning_rate, (moment_w/(np.sqrt(RMS_w)+self.epsilon)))  # for weights
            layer.biases = layer.biases - self.learning_rate*(moment_b/(np.sqrt(RMS_b)+self.epsilon))  # for biases

    def Adam_Optimization(self, t, dw, db):
        # momentum - beta1
        self.momentum_w = np.dot(self.beta1, self.momentum_w) + np.dot((1-self.beta1), dw)  # for weights
        self.momentum_b = np.dot(self.beta1, self.momentum_b) + np.dot((1-self.beta1), db)  # for biases

        # RMSProp - beta2
        self.RMSProp_w = np.dot(self.beta2 * self.RMSProp_w) + np.dot((1 - self.beta2), (dw**2))  # for weights
        self.RMSProp_b = np.dot(self.beta2 * self.RMSProp_b) + np.dot((1 - self.beta2), (db**2))  # for biases

        # bias correction
        moment_w = np.dot(self.momentum_w, 1 / (1-self.beta1**t))  # correct momentum for weights
        moment_b = np.dot(self.momentum_b, 1 / (1-self.beta1**t))  # correct momentum for biases
        RMS_w = np.dot(self.RMS_w,  1 / (1-self.beta2**t))  # correct RMSProp for weights
        RMS_b = np.dot(self.RMS_b, 1 / (1-self.beta2**t))  # correct RMSProp for biases

        # update params
        weight_change = np.dot(self.learning_rate, (moment_w/(np.sqrt(RMS_w)+self.epsilon)))
        bias_change = self.learning_rate*(moment_b/(np.sqrt(RMS_b)+self.epsilon))

        return weight_change, bias_change



