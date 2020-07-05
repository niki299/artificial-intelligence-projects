
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.preprocessing import image
import pandas as pd
import os

def get_data(path):
    csv = pd.read_csv(path + '\\metadata\\chest_xray_metadata.csv')

    normal = []
    virus = []
    bacteria = []

    for row in csv.values:

        if 'Normal' in row[2:]:
            normal.append(row)
        elif 'Virus' in row[2:]:
            virus.append(row)
        elif 'bacteria' in row[2:]:
            bacteria.append(row)

    return normal, virus, bacteria

def save(path, folder, label, file):

    os.rename(path + '\\' + file,
              path + '\\' + folder + '\\' + label + '\\' + file)

def divide(path, list, label, ratio):

    cnt = 0

    for row in list:
        # Get 'ratio' amount of data as a training data
        if cnt < np.floor(ratio * len(list)):
            save(path, 'training', label, str(row[1]))
        # The rest is for validation
        else:
            save(path, 'validation', label, str(row[1]))

        cnt += 1

def create_folders(path):

    if not os.path.exists(path + '\\training'):
        os.makedirs(path + '\\training')
        os.makedirs(path + '\\training\\Normal')
        os.makedirs(path + '\\training\\Virus')
        os.makedirs(path + '\\training\\bacteria')

    if not os.path.exists(path + '\\validation'):
        os.makedirs(path + '\\validation')
        os.makedirs(path + '\\validation\\Normal')
        os.makedirs(path + '\\validation\\Virus')
        os.makedirs(path + '\\validation\\bacteria')

def split_data(path):

    normal, virus, bacteria = get_data(path)
    create_folders(path)

    divide(path, normal, 'Normal', 0.8)
    divide(path, virus, 'Virus', 0.8)
    divide(path, bacteria, 'bacteria', 0.8)

def main():

    path = 'C:\\Faks\\3. Godina\\6. Semestar\\ORI\\pacman_project\\chest_xray_data_set'

    # This function split data into training and validation folder
    # You call it only when you are using this code for the first time
    # split_data(path)

    model = keras.models.Sequential([

        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        keras.layers.MaxPooling2D(2, 2),

        # The second convolution
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        # The third convolution
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        # The fourth convolution
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        # The fifth convolution
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),  # 512 neuron hidden layer
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('Normal') clas and 1 for ('pneumonia') class
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # to get the summary of the model
    model.summary()

    # configure the model for traning by adding metrics
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    training_datag_gen = ImageDataGenerator(rescale=1 / 255)
    validation_data_gen = ImageDataGenerator(rescale=1 / 255)

    training_data = training_datag_gen.flow_from_directory(
        path + '\\training',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    validation_data = validation_data_gen.flow_from_directory(
        path + '\\validation',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary'
    )

    # training the model
    history = model.fit(
        training_data,
        steps_per_epoch=10,
        epochs=10,
        validation_data = validation_data
    )


if __name__ == '__main__':
    main()