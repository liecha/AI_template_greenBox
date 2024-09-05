# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from neural_network_rnn import RNN
from neural_network_transformer import Transformer, transformer_encoder
from parameters import glodbal_settings

# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt 

class Model:
    def __init__(self, dataset, agent, param):
        
        # Training parameters
        self.epochs = glodbal_settings['epochs']
        self.batch_size = glodbal_settings['batch_size']
        
        # Project parameters
        self.dataset = dataset
        self.len_input = glodbal_settings['len_input']
        self.prediction_win = glodbal_settings['prediction_window']
        self.len_pred = glodbal_settings['len_pred']
        self.agent = agent
        self.param = param
        self.simulation_test = []
        self.data = []
        self.data_agent = []
        self.df_split_train_test = []
        self.scaler = []
        self.scaled_data = []
        self.X_train = np.array
        self.y_train = np.array
        self.X_test = np.array
        self.y_test = np.array
        self.y_pred = np.array
        self.y_pred_inv = np.array
        self.y_test_inv = np.array
        self.scaled_data_pred = []
        
        
    def data_split_80_20(self):
        dataset = self.dataset
        limit = int(len(dataset)*0.8)
        df_training_set = dataset.iloc[0:limit]
        df_test_set = dataset.iloc[limit:-1]
        self.df_split_train_test.append(df_training_set)
        self.df_split_train_test.append(df_test_set)
        
        print('Dataset split into training and testdata...')
        print(df_training_set)       
        print(df_test_set)
        

    def scaling_data(self):
        
        # Feature Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = []
        self.scaler.append(scaler)
        
        # Save the scaler to a file
        pickle.dump(scaler, open(glodbal_settings['scaler_path'], 'wb'))
        
        print('Scaling data for verification phase...')
        data_train = self.df_split_train_test[0][self.param].values.reshape(-1, 1)
        data_test = self.df_split_train_test[1][self.param].values.reshape(-1, 1)

        scaled_data_train = scaler.fit_transform(data_train)
        self.scaled_data.append(scaled_data_train)
        
        scaled_data_test = scaler.fit_transform(data_test)
        self.scaled_data.append(scaled_data_test)
        print('Data for parameter ' + self.param + ' is scaled...')


    def series_to_supervised(self):
        dataset_scaled = self.scaled_data[0]
        X_train = []
        y_train = []
        
        len_trainset = len(dataset_scaled)
        for i in range(self.len_input, len_trainset - self.prediction_win + 1):
            X_train.append(dataset_scaled[i - self.len_input:i,0])
            y_train.append(dataset_scaled[i + self.prediction_win - 1:i + self.prediction_win,0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        self.X_train = X_train
        self.y_train = y_train
        print('   - shape supervised serie == ', self.X_train.shape, self.y_train.shape)
        print('   - total dataset lenght   == ', len(dataset_scaled))
        print('   - intervall range        == ' + str(self.len_input) + ' --> ' + str(len_trainset - self.prediction_win + 1))
        print('   - no of X batches        == ', (len_trainset - self.prediction_win + 1) - self.len_input)
        print('   - data in each batch     == ', self.len_input)


    def prepare_verification_data(self):
        dataset_scaled_test = self.scaled_data[1]
        
        X_test = []
        y_test = []
        
        for i in range(self.len_input, self.len_input + self.len_pred + 1):      
            X_test.append(dataset_scaled_test[i-self.len_input:i, 0])
            y_test.append(dataset_scaled_test[i + self.prediction_win - 1:i + self.prediction_win,0])
            
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))        
        self.X_test = X_test
        self.y_test = y_test
        print('   - shape verification data == ', self.X_test.shape, self.y_test.shape)
        print('   - intervall range         == ' + str(self.len_input) + ' --> ' + str(self.len_input + self.len_pred + 1))
        print('   - predicted no of value   == ', (self.len_input + self.len_pred + 1) - self.len_input)
        print('   - data in each batch      == ', self.len_input)
    
        
    def train_predict_verification(self, network):
        if network == 'rnn':
            print('Predparing neural network (RECURRENT Neural Network...)')
            neural_net = RNN(self.X_train)
        if network == 'transformer':
            print('Predparing neural network (TRANSFORMER Neural Network...)')
            neural_net = Transformer(self.X_train)
        history = neural_net.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.X_test, self.y_test), verbose=2, shuffle=False)
        #self.plot_training_history(history, 0)
        scaler = self.scaler[0]
        self.y_pred = neural_net.predict(self.X_test)
        self.y_pred = self.y_pred.reshape(-1,1)
        self.y_pred_inv = scaler.inverse_transform(self.y_pred)
        self.y_test_inv = scaler.inverse_transform(self.y_test)

        
    # Hyper Parameter Evaluation
    # SOURCE: https://medium.com/analytics-vidhya/hypertuning-a-lstm-with-keras-tuner-to-forecast-solar-irradiance-7da7577e96eb
    def build_model_rnn(self, hp):
        X_train = self.X_train
        y_train = self.y_train
        model = Sequential()
        model.add(LSTM(hp.Int(name='input_unit', min_value=32 ,max_value=512, step=32), return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        for i in range(hp.Int('n_layers', 1, 4)):
            model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
        model.add(LSTM(hp.Int('layer_2_neurons', min_value=32, max_value=512, step=32)))
        model.add(Dropout(hp.Float('Dropout_rate', min_value=0, max_value=0.5, step=0.1)))
        model.add(Dense(y_train.shape[1], activation=hp.Choice('dense_activation', values=['relu', 'sigmoid'], default='relu')))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])
        return model


    def build_model_transformer(self, hp):     
        
        '''
        Building a transformer model with hyer-parameter to explore
        Hyper-parameters to set:
          1. emb_vector_size: Embedding Vector Dimensions,   Number of dimensions for the embedding positional vector
          2. num_heads: Number of heads
          3. ff_dim:  Hidden layer size in feed forward network inside transformer
          4. num_transformer_blocks: Total number of encoder layers
          5. num_mlp_layers: Number of mlp layers after the encoder layer
          6. mlp_units: Number of units in the Dense networks
          
        '''
        X_train = self.X_train
        input_shape = (X_train.shape[1], X_train.shape[2])
        inputs = keras.Input(shape=input_shape)
      
        # Hyper-parameters
        emb_vector_size = hp.Int('EVD', min_value=100, max_value=250, step=50)
        num_heads = hp.Int('num_heads', min_value=4, max_value=6, step=1)
        ff_dim = hp.Int('ff_dim', min_value=32, max_value=64, step=32)
        num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=2, max_value=4, step=1)
      
        # Other Hyper-parameters to explore 
        num_mlp_layers = 1
        mlp_units = hp.Int('ff_dim', min_value=32, max_value=64, step=32)
      
        mlp_dropout = 0.3
        dropout=0.2
      
        x = inputs
              
        for _ in range(num_transformer_blocks):
            x = transformer_encoder(x, emb_vector_size, num_heads, ff_dim, dropout)
      
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
      
        mlp_dropout = 0.4
        
        for dim in range(num_mlp_layers):
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1)(x)
        transformer_model = keras.Model(inputs, outputs)
        transformer_model.compile(loss='sparse_categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy', 
                                ])   
        return transformer_model
    
    
    def tuning_model_rnn(self):        
        tuner= RandomSearch(
            self.build_model_rnn,
            objective='mse',
            max_trials=2,
            executions_per_trial=1
            )
        
        tuner.search(
            x = self.X_train,
            y = self.y_train,
            epochs=100,
            batch_size=128,
            validation_data=(self.X_test, self.y_test),
        )
        
        best_model = tuner.get_best_models(num_models=1)[0]
        #best_model.save('BestLSTMModel.h5')
        scaler = self.scaler[0]
        self.y_pred = best_model.predict(self.X_test[0].reshape((1, self.X_test[0].shape[0], self.X_test[0].shape[1])))
        print(self.y_pred)
        self.y_pred = self.y_pred.reshape(-1,1)
        self.y_pred_inv = scaler.inverse_transform(self.y_pred)
        self.y_test_inv = scaler.inverse_transform(self.y_test)


    def tuning_model_transformer(self):   
        # Keras Tuner Stratergy 
        tuner = kt.Hyperband(self.build_model_transformer,
                             objective='val_accuracy',
                             max_epochs=100,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')
        
        # Early Stopping 
        ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',    
                                              patience=4,    
                                              verbose=1,    
                                              restore_best_weights='True',
                                              min_delta = 0.1
                                             )

        tuner.search(
            x = self.X_train,
            y = self.y_train,
            epochs=100,
            batch_size=128,
            validation_data=(self.X_test, self.y_test),
            callbacks=[ES]
        )
        
        best_model = tuner.get_best_models(num_models=1)[0]
        #best_model.save('BestLSTMModel.h5')
        scaler = self.scaler[0]
        self.y_pred = best_model.predict(self.X_test[0].reshape((1, self.X_test[0].shape[0], self.X_test[0].shape[1])))
        print(self.y_pred)
        self.y_pred = self.y_pred.reshape(-1,1)
        self.y_pred_inv = scaler.inverse_transform(self.y_pred)
        self.y_test_inv = scaler.inverse_transform(self.y_test)
        
        
    def plot_training_history(self, history, stearing_index):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('Training results history ' + ' (win ' + str(stearing_index + 1) + ')')
        plt.legend()
        plt.show()


    def plot_result_verification(self):    
            fig = plt.figure()
            ax = fig.add_axes([0,0,1,1])
            
            title = ""
            label = ""
            
            if self.param == 'longitude':
                title = 'Prediction LONGITUDE ' + self.agent
                label = 'LONGITUDE' 
            if self.param == 'latitude':
                title = 'Prediction LATITUDE ' + self.agent
                label = 'LATITUDE'
            
            ax.plot(self.y_pred_inv)
            ax.plot(self.y_test_inv)
            
            ax.grid(True)
            fig.autofmt_xdate()            
            plt.title(title)
            plt.xlabel('Timeslot')
            plt.ylabel(label)
            plt.legend(['Prediction', 'Real value'])
