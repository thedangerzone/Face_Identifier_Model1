import PIL.Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp
import datetime
#from skimage.util import random_noise


def Get_Images(path, test_size):
    os.chdir(path)

    images, people = [], []
    for i, user in enumerate(os.listdir()[:]):
        for file in os.listdir(path + f'\\{user}')[:]:
            image = PIL.Image.open(path + f'\\{user}\\' + file)
            image = np.array(image.convert('L'))
            image = face_detection(image)
            #print(image.shape)
            if image.shape == (1, 128, 128):
                image = np.reshape(image,(128, 128))
                images.append(image)
                people.append(i)

    inp_train, inp_test, label_train, label_test = train_test_split(images, people, test_size=test_size)

    inp_final_train = []
    labels_final_train = []
    inp_final_test = []
    labels_final_test = []

    for inp in range(0, len(inp_train)-1, 2):
        inp_final_train.append(np.append(inp_train[inp], inp_train[inp+1], axis=0))
        labels_final_train.append(int(label_train[inp] == label_train[inp+1]))

    for inp in range(0, len(inp_test)-1, 2):
        inp_final_test.append(np.append(inp_test[inp], inp_test[inp+1], axis=0))
        labels_final_test.append(int(label_test[inp] == label_test[inp+1]))

    inp_final_train, inp_final_test, labels_final_train, labels_final_test = np.array(inp_final_train), np.array(inp_final_test), \
                                       np.array(labels_final_train), np.array(labels_final_test)

    return inp_final_train, inp_final_test, labels_final_train, labels_final_test

def face_detection(image):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    #cap = cv2.VideoCapture(0)
    #ret, frame = cap.read()

    faces = []
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as face_detection:
        count = 0
        fails_Count = 0
        if True:
            #print(image.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            h, w, c = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                xmin = int(results.detections[0].location_data.relative_bounding_box.xmin * w)
                ymin = int(results.detections[0].location_data.relative_bounding_box.ymin * h)
                width = int(results.detections[0].location_data.relative_bounding_box.width * w)
                height = int(results.detections[0].location_data.relative_bounding_box.height * h)

                image = image[ymin:ymin+height, xmin:xmin+width]
                try:
                    image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                except cv2.error:
                    fails_Count+=1

                faces.append(image)
                count+=1
                #cv2.imshow('Face Detection', image)
                #cv2.waitKey(0)
            else:
                return np.array([False])

        return np.array(faces)

def scale_data(inp_train, inp_test = None):

    m = np.max(inp_train)

    inp_train = inp_train / m
    try:
        if inp_test is not None:
            inp_test = inp_test / m
    except:
        inp_test = inp_test / m
    return  inp_train, inp_test

def Create_Data_With_Noise(path,n, mode = 'gaussian',std = 7, p = 0.95):
    l = os.listdir(path)
    for i,image in enumerate(l):
        for _ in range(n):
            if mode == 'gaussian':
                img = np.array(PIL.Image.open(path+'\\'+image))
                # Add salt-and-pepper noise to the image.
                noise_img = np.random.normal(0,std,img.shape) * np.random.binomial(1,p,img.shape)
                noise_img = img + noise_img
                img = PIL.Image.fromarray(noise_img).convert('L')
                img.save(path+'\\'+'noise'+image)
            else:
                img = np.array(PIL.Image.open(path+'\\'+image))
                # Add salt-and-pepper noise to the image.
                noise_img = np.random.binomial(1,p,img.shape)
                noise_img = img * noise_img
                img = PIL.Image.fromarray(noise_img).convert('L')
                img.save(path+'\\'+f'noise{str(datetime.datetime.now().microsecond)}'+image)
        if i % 50 ==0:
            print(f'created {n} new images from the {i} image')







