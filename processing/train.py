# for training the datase
import json
import tensorflow.keras as keras
from keras.applications.densenet import layers

from preprocess import generating_training_sequences, sequence_length, mapping_path

LOSS = "sparse_categorical_crossentropy"
# error function for training
LEARNING_RATE = 0.001
EPOCHS = 50
NUM_UNITS = [256]
# number of units of neuron in internal layer of network
# this is one layer with 256 neurons
BATCH_SIZE = 64
# amount of samples it will see before running back propogation
save_model_path = "model.h5"


# .h5 for saving keras model


def output_units_(mapping_path):
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)
    return len(mappings.keys())


def build_model(output_units, num_units, loss, learning_rate):
    # create model architecture
    input = keras.layers.Input(shape=(None, output_units))
    # shape is of the data which we will input in our model
    # None enables us to have as many timestamps as we want ,we than can generate however melodies we want
    # output_units tells us how many elements we have for each timestamp i.e our vocabulary size
    x = keras.layers.LSTM(num_units[0])(input)
    # add another node to graph

    x = keras.layers.Dropout(0.2)(x)
    # to avoid overfitting we use dropout layer
    # here the rate refers to amount of input data to be dropped
    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model

    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
    model.summary()
    # print some information about all the model
    return model


def train(output_units=output_units_(mapping_path), num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    # it will do all the functions that will train our dataset
    # generate training sequences
    # it is already created in preprocess.py
    input, target = generating_training_sequences(sequence_length)
    # print(type(input))
    # print(input.ndim)
    # print(type(target))
    # build the network
    model = build_model(output_units, num_units, loss, learning_rate)
    # output units i.e how many neurons are there in output layer
    # train the model
    model.fit(input, target, epochs=EPOCHS, batch_size=BATCH_SIZE)
    # save the model for future uses
    model.save(save_model_path)


if __name__ == '__main__':
    train()
