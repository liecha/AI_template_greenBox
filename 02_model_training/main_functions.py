# -*- coding: utf-8 -*-
# Import own modules
import pandas as pd

# Import own modules
from model import Model
from parameters import glodbal_settings

   
def run_AI_waraps_training(agents): 
    params = glodbal_settings['params']
    dataset_agent = pd.read_csv(glodbal_settings['data_path'])
    for agent in agents:
        for i in range(0, len(params)):           
            model = Model(dataset_agent, agent, params[i])
            print('---------------- PROCESSING ' + params[i] + ' ----------------')      
            training_phase(model)                    
    print('MODEL TRAINING COMPLETED...')

            
def training_phase(model):        
    model.scaling_data()
    model.series_to_supervised()
    model.train_model(glodbal_settings['network'])