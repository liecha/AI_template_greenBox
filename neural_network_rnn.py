# -*- coding: utf-8 -*-

# Buildning the RNN (Recurrent Neural Network)
# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def RNN(X_train):
    
    '''
    Hyper parameter tuning results
    Search: Running Trial #2
    Value             |Best Value So Far |Hyperparameter
    384               |32                |input_unit
    2                 |3                 |n_layers
    416               |64                |lstm_0_units
    416               |32                |layer_2_neurons
    0.1               |0                 |Dropout_rate
    relu              |relu              |dense_activation
    416               |32                |lstm_1_units
    256               |32                |lstm_2_units
    

    regressor = Sequential()
    input_unit = 32
    lstm_0_units = 64
    lstm_1_units = 32
    lstm_2_units = 32
    layer_2_neurons = 32
    dropout = 0
    dense_activation = 'relu'
    regressor.add(LSTM(units=input_unit, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(LSTM(units=lstm_0_units, return_sequences=True))
    regressor.add(LSTM(units=lstm_1_units, return_sequences=True))
    regressor.add(LSTM(units=lstm_2_units, return_sequences=True))
    regressor.add(LSTM(units=layer_2_neurons))
    regressor.add(Dropout(dropout))
    regressor.add(Dense(1)) #regressor.add(Dense(1, activation=dense_activation))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor
    
    '''
    neurons = 512
    regressor = Sequential()

    regressor.add(LSTM(units=neurons, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])))

    # Dropout regulation (use to be 20%)
    # --> this is the number of neurons to be ignored
    # First layer
    regressor.add(Dropout(0.2))

    # Second layer
    regressor.add(LSTM(units=neurons, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Third layer
    regressor.add(LSTM(units=neurons, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Forth layer - last layer before output  layer
    regressor.add(LSTM(units=neurons))
    regressor.add(Dropout(0.2))

    # Output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    # NOTE: Optimizer - RMSprops is recommended for RNN but Adam was detected to
    # be a better choice for this problem
    # mean_squared_error
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    return regressor