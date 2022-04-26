import model
import learn
import Layer
import LoadImages
import os
import numpy as np

def main():
    inp_train, inp_test, label_train, label_test = LoadImages.Get_Images(
                                                    path=r'C:\Users\omers\Desktop\model project\anshim',
                                                    test_size=0.2)

    inp_train = np.float16(inp_train)
    inp_test = np.float16(inp_test)
    label_train = np.float16(label_train)
    label_test = np.float16(label_test)

    inp_train, inp_test = LoadImages.scale_data(inp_train, inp_test)

    print(inp_train.shape, label_train[0].shape)

    print(np.max(label_train) + 1)

    layer1 = Layer.Layer(neuron_amount=256, input_amount=32768, activation='relu')
    layer2 = Layer.Layer(neuron_amount=128, input_amount=256, activation='relu')
    layer3 = Layer.Layer(neuron_amount=128, input_amount=128, activation='relu')
    layer4 = Layer.Layer(neuron_amount=128, input_amount=128, activation='relu')
    layer5 = Layer.Layer(neuron_amount=1, input_amount=128, activation='sigmoid')

    my_model = model.Model()

    my_model.Add_Layer(layer1)
    my_model.Add_Layer(layer2)
    my_model.Add_Layer(layer3)
    my_model.Add_Layer(layer4)
    my_model.Add_Layer(layer5)

    inp_train = np.reshape(inp_train, (2146, 128*256))
    print('data loaded!')
    print('input shape', inp_test.shape)
    print('label shape', label_train.shape)

    ch_path = os.environ.get('userprofile')+r'\Documents\Model'
    my_model.train(inp_train, label_train, inp_test, label_test,
                epochs=300,print_every=5, model_save_location=ch_path)

    print('model trained!')


if __name__ == '__main__':
    main()