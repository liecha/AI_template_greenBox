# -*- coding: utf-8 -*-

import keras 
from keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def Transformer(X_train):
    '''
    Hyper parameter tuning results
    Search: Running Trial #68

    Value             |Best Value So Far |Hyperparameter
    150               |150               |EVD (embedde vector size/dimension)
    5                 |6                 |num_heads
    64                |32                |ff_dim
    4                 |2                 |num_transformer_blocks
    2                 |2                 |tuner/epochs
    0                 |0                 |tuner/initial_epoch
    4                 |4                 |tuner/bracket
    0                 |0                 |tuner/round
    '''
    input_shape = (X_train.shape[1], X_train.shape[2])
    head_size = 150
    num_heads = 6
    ff_dim = 32
    num_transformer_blocks = 2
    mlp_units = [1024]
    mlp_dropout = 0.3
    dropout = 0.2

    inputs = keras.Input(shape=input_shape)
    x = inputs

    for _ in range(num_transformer_blocks):
        transformer_encoder(
            x, head_size, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

