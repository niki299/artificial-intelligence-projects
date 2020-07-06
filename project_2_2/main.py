
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.models import model_from_json
import pandas as pd
import os
from matplotlib import pyplot
import random
from keras.callbacks import ModelCheckpoint


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

    random.shuffle(normal)
    random.shuffle(virus)
    random.shuffle(bacteria)

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

def make_subdirs(path):

    if not os.path.exists(path + '\\Normal'):
        os.makedirs(path + '\\Normal')

    if not os.path.exists(path + '\\Virus'):
        os.makedirs(path + '\\Virus')

    if not os.path.exists(path + '\\bacteria'):
        os.makedirs(path + '\\bacteria')


def create_folders(path):

    if not os.path.exists(path + '\\training'):
        os.makedirs(path + '\\training')
        make_subdirs(path + '\\training')

    if not os.path.exists(path + '\\validation'):
        os.makedirs(path + '\\validation')
        make_subdirs(path + '\\validation')

def split_data(path):

    normal, virus, bacteria = get_data(path)
    create_folders(path)

    divide(path, normal, 'Normal', 0.8)
    divide(path, virus, 'Virus', 0.8)
    divide(path, bacteria, 'bacteria', 0.8)

def split_test(path):

    csv = pd.read_csv(path + '\\chest_xray_test_dataset.csv')
    make_subdirs(path + '\\test')

    for row in csv.values:
        for label in row[2:]:
            print(label)
            if label == 'Normal' or label == 'Virus' or label == 'bacteria':
                os.rename(path + '\\' + 'test' + '\\' + str(row[1]),
                          path + '\\' + 'test' + '\\' + label + '\\' + str(row[1]))

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    return model

def training(model, train_path):

    training_data_gen = ImageDataGenerator(rescale=1 / 255)
    validation_data_gen = ImageDataGenerator(rescale=1 / 255)

    training_data = training_data_gen.flow_from_directory(
        train_path + '\\training',
        target_size=(300, 300),
        batch_size=128,
        class_mode='categorical'
    )

    validation_data = validation_data_gen.flow_from_directory(
        train_path + '\\validation',
        target_size=(300, 300),
        batch_size=128,
        class_mode='categorical'
    )

    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # training the model
    history = model.fit(
        training_data,
        steps_per_epoch=10,
        epochs=10,
        validation_data=validation_data,
        callbacks=callbacks_list
    )

    # ploting training metrics
    pyplot.subplot(1,2,1)
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.subplot(1,2,2)
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.show()

    return model

def init_model():

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
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    # to get the summary of the model
    model.summary()

    # configure the model for traning by adding metrics
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

    return model

def main():


    train = True

    train_path = 'C:\\Faks\\3. Godina\\6. Semestar\\ORI\\pacman_project\\chest_xray_data_set'
    test_path = 'C:\\Faks\\3. Godina\\6. Semestar\\ORI\\pacman_project\\chest-xray-dataset-test'

    # This function split data into training and validation folder
    # You call it only when you are using this code for the first time
    # split_data(train_path)
    # split_test(test_path)

    model = init_model()
    if train:
        model = training(model, train_path)
    else:
        model.load_weights('weights.best.hdf5')

        # configure the model for traning by adding metrics
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])


    # loading test data
    test_data_gen = ImageDataGenerator(rescale=1 / 255)

    test_data = test_data_gen.flow_from_directory(
        test_path + '\\test',
        target_size=(300, 300),
        batch_size=128,
        class_mode='categorical'
    )

    results = model.evaluate_generator(test_data, 640)

    print('Accuracy on test data: ', results[1])
    print('Loss on test data: ', results[0])


if __name__ == '__main__':
    main()