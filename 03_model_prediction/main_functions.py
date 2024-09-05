# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
from model import Model
from parameters import glodbal_settings
from keras.models import load_model


def loading_process():
    network_batch = []
    params = glodbal_settings['params']
    for param in params:
        loaded_model = load_network(param)
        network_batch.append(loaded_model)
    return network_batch
 
   
def run_AI_waraps(mqtt): 
    print('Network loading process...')
    network_batch = loading_process()
    print('Network loading process completed...')
    params = glodbal_settings['params']
    agents = glodbal_settings['agents']
    while True:
        for agent in agents:            
            prediction_storage = { 'latitude': [], 'longitude': [],'altitude': [] }
            for i in range(0, len(params)):           
                model = Model(agent, mqtt, params[i])
                print('------------- PROCESSING ' + params[i] + ' -------------')      
                param_result = prediction_phase(model, network_batch[i]) 
                if params[i] == 'lon':
                    prediction_storage['longitude'] = param_result
                if params[i] == 'lat':
                    prediction_storage['latitude'] = param_result
                if params[i] == 'alt':
                    prediction_storage['altitude'] = param_result
            publish_format = model.publish_reshape(prediction_storage)
            smooth_point = model.smoothening(publish_format)
            model.create_sensor_payload(publish_format[len(publish_format)-1:])
            #model.create_sensor_payload(smooth_point)
    print('Prediction published...')
    

def load_network(param):
    if glodbal_settings['network'] == 'rnn':
        network_name = glodbal_settings['model_path_rnn'] + param
        loaded_model = load_model(network_name)
        print('The pre-trained RNN network is loaded...')
        print(network_name)
    if glodbal_settings['network'] == 'transformer':
        network_name = glodbal_settings['model_path_transformer'] + param
        loaded_model = load_model(network_name)
        print('The pre-trained TRANSFORMER network is loaded...')
        print(network_name)
    return loaded_model

            
def prediction_phase(model, loaded_network): 
    model.get_latest_positions()
    model.scaling_data()
    model.series_to_supervised()
    future_inv = model.make_prediction(loaded_network)
    return future_inv

    



    
