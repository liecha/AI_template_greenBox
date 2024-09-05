# -*- coding: utf-8 -*-
# Import libraries
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import pickle
from parameters import glodbal_settings

class Model:
    def __init__(self, agent, mqtt, param):
        
        # Training parameters
        self.epochs = glodbal_settings['epochs']
        self.batch_size = glodbal_settings['batch_size']
        
        # Project parameters
        self.len_input = glodbal_settings['len_input']
        self.prediction_win = glodbal_settings['prediction_window']
        self.len_pred = glodbal_settings['len_pred']
        self.agent = agent
        self.mqtt = mqtt
        self.param = param
        self.agentscache = mqtt.agentscache
        self.data = []
        self.data_agent = []
        self.scaler = []
        self.scaled_data = []
        self.X_train = np.array
        self.y_train = np.array
        self.X_test = np.array
        self.y_pred = np.array
        self.y_pred_inv = np.array
  
           
    def get_latest_positions(self):
        no_of_points = self.len_input + self.len_pred + 1
        print('Selecting the latest ' + str(no_of_points) + ' positions...')
        dataframe = self.agentscache.get_MQTT_Data_CSV(self.agent, no_of_points)
        dataframe.to_csv(glodbal_settings['data_path_pred'], index=False)       
        print('The last ' + str(no_of_points) + ' positions are saved to the project folder...')
    
    
    def scaling_data(self):
        # Feature Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.append(scaler)
       
        # Save the scaler to a file
        pickle.dump(scaler, open(glodbal_settings['scaler_path_pred'], 'wb'))
        
        data_test = pd.read_csv(glodbal_settings['data_path_pred'])[self.param].values.reshape(-1, 1)
        scaled_data_test = scaler.fit_transform(data_test)
        self.scaled_data.append(scaled_data_test)
        print('Data for parameter ' + self.param + ' is scaled...')
           
         
    def series_to_supervised(self):
        dataset_scaled_test = self.scaled_data[0]
        
        X_test = []
        
        for i in range(self.len_input, self.len_input + self.len_pred + 1):      
            X_test.append(dataset_scaled_test[i-self.len_input:i, 0])
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))        
        self.X_test = X_test
        print('   - shape verification data == ', self.X_test.shape)
        print('   - intervall range         == ' + str(self.len_input) + ' --> ' + str(self.len_input + self.len_pred + 1))
        print('   - predicted no of value   == ', (self.len_input + self.len_pred + 1) - self.len_input)
        print('   - data in each batch      == ', self.len_input)

    
    def make_prediction(self, loaded_model): 
        scaler = self.scaler[0]
        self.y_pred = loaded_model.predict(self.X_test)
        self.y_pred = self.y_pred.reshape(-1,1)
        self.y_pred_inv = scaler.inverse_transform(self.y_pred)      
        return self.y_pred_inv.flatten().tolist()
        '''
        for i in range(future_pred_count):
            prediction = loaded_model.predict(currentStep) # set dimentions
            future.append(prediction[0][0]) 
            currentStep = np.append(currentStep[1:], prediction[0][0], axis=None ) #store the future steps
            
            # Inverse the scaling of the prediction
            prediction = prediction.reshape(-1, 1)
            scaler = self.scaler[0]
            prediction_inv = scaler.inverse_transform(prediction)
            prediction_inv.flatten().tolist()
            future_inv.append(prediction_inv)
        return future_inv
        
        self.y_pred = loaded_model.predict(self.X_test)
        self.y_pred = self.y_pred.reshape(-1, 1)
        scaler = self.scaler[0]
        self.y_pred_inv = scaler.inverse_transform(self.y_pred)       
        print('---------------- 3.) PREDICTION MADE AND SAVED FOR ' + self.param + ' ----------------') 
        return self.y_pred_inv.flatten().tolist()
        '''

    def publish_reshape(self, prediction):
        publish_format = []
        for i in range(0, len(prediction['latitude'])):
            publish_format.append({
                'latitude': prediction['latitude'][i],
                'longitude': prediction['longitude'][i],
                'altitude': prediction['altitude'][i]
            })
        return publish_format
    
    
    def smoothening(self, publish_format):
        X = []
        Y = []
        Z = []
        for i in range(0, len(publish_format)):
            X.append(publish_format[i]['longitude'])
            Y.append(publish_format[i]['latitude'])
            Z.append(publish_format[i]['altitude'])
        last_X = max(X)
        a, b = self.best_fit(X, Y)
        #self.plot_scatter_line(X, Y, a, b, last_X)
        y_smooth = a + b * last_X
        smooth_point = [{ 'latitude': y_smooth, 'longitude': last_X,'altitude': Z[-1] }]
        return smooth_point
        

    def best_fit(self, X, Y):
        xbar = sum(X)/len(X)
        ybar = sum(Y)/len(Y)
        n = len(X)
    
        numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
        denum = sum([xi**2 for xi in X]) - n * xbar**2
    
        b = numer / denum
        a = ybar - b * xbar
    
        #print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))
        return a, b
    
    
    def plot_scatter_line(self, X, Y, a, b, last_X):
        plt.figure()
        plt.scatter(X, Y)
        yfit = [a + b * xi for xi in X]
        plt.plot(X, yfit)
        y_smooth = a + b * last_X
        plt.plot(last_X, y_smooth, marker="o", markersize=20, markeredgecolor="red")
 
           
    def create_sensor_payload(self, final_result):
        #topic = self.agentscache.get_topic(self.agent)
        poi_topic = 'waraps/unit/air/simulation/' + self.agent + '/sensor/prediction'
        self.mqtt.publish(poi_topic, final_result, retain=False)
        #self.mqtt.disconnect()