from keras.models import Sequential
from keras.layers import Dense
import numpy


class simple_dnn:
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build_simple_nn_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_simple_2layer_dnn_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

