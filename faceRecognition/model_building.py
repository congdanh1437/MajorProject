import numpy as np
import os
from PIL import Image

TRAIN_DATA = 'datasets/train_data'
TEST_DATA = 'datasets/test_data'
Xtrain = []
Ytrain = []

# Xtrain = [x[0] for i, x in enumerate(Xtrain)]

Xtest = []
Ytest = []
dict = {'trainCongDanh': [1, 0, 0, 0, 0], 'testCongDanh': [1, 0, 0, 0, 0]}

def getData(dirData, lstData):
    for whatever in os.listdir(dirData):
        whatever_path = os.path.join(dirData,whatever)
        lst_filename_path = []
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)
            label = filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            lst_filename_path.append((img, dict[label]))

        lstData.extend(lst_filename_path)
    return lstData

Xtrain = getData(TRAIN_DATA, Xtrain)
Xtest = getData(TEST_DATA, Xtest)

for i in range(3):
    np.random.shuffle(Xtrain)

from keras import layers
from keras import models

model_training_first = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(148, 148, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000,activation='relu'),
    layers.Dense(10,activation='softmax')
])

# model_training_first.summary()

model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

model_training_first.fit(np.array([x[0] for _, x in enumerate(Xtrain)]), np.array([y[1] for _, y in enumerate(Xtrain)]), epochs=10)

model_training_first.save('model_faceRecognition.h5')
models = models.load_model('model_faceRecognition.h5')