# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
import numpy as np
import pickle
from neural_network_rnn import RNN
from neural_network_transformer import Transformer
from parameters import glodbal_settings

class Model:
    def __init__(self, dataset_agent, agent, param):
        
        # Training parameters
        self.epochs = glodbal_settings['epochs']
        self.batch_size = glodbal_settings['batch_size']
        
        # Project parameters
        self.dataset_agent = dataset_agent
        self.len_input = glodbal_settings['len_input']
        self.prediction_win = glodbal_settings['prediction_window']
        self.len_pred = glodbal_settings['len_pred']
        self.agent = agent
        self.param = param
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
        self.scaled_data_pred = []
        
                
    def scaling_data(self):
        scaler = pickle.load(open('../saved_scaler/scaler.sav', 'rb'))      
        data = self.dataset_agent[self.param].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)
        self.scaled_data.append(scaled_data)
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

    
    def train_model(self, network):
        if network == 'rnn':
            print('Predparing neural network (RECURRENT Neural Network...)')
            neural_net = RNN(self.X_train)
            neural_net.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=False)    
            network_name = '../saved_model/RNN_model_' + self.param
            neural_net.save(network_name)
            print('Model saved --> ' + network_name)
        if network == 'transformer':
            print('Predparing neural network (TRANSFORMER Neural Network...)')
            neural_net = Transformer(self.X_train)
            neural_net.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2, shuffle=False)    
            network_name = '../saved_model/transformer_model_' + self.param
            neural_net.save(network_name)
            print('Model saved --> ' + network_name)
        