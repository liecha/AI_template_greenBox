# -*- coding: utf-8 -*-
# Import own modules
import pandas as pd

# Import own modules
from model import Model
from parameters import glodbal_settings


def run_AI_verification(agents): 
    params = glodbal_settings['params']
    dataset = pd.read_csv(glodbal_settings['data_path'])        
    for agent in agents:
        for i in range(0, len(params)):           
            model = Model(dataset, agent, params[i])
            print('Preparing dataset...')
            print('---------------- START PROCESSING PARAMETER ' + params[i] + ' ----------------')           
            print('Entering verification phase...')  
            verification_phase(model)   
    print('VERIFICATION PROCESS COMPLETED...')                       
    

def verification_phase(model):
    print('Split dataset 80/20 training/test...')
    model.data_split_80_20()
    print('Scaling process started...')
    model.scaling_data()
    print('Reshaped data for supervised learning...')
    model.series_to_supervised()
    print('Prepare data for test and verification...')
    model.prepare_verification_data()
    print('Training and verification process in action...')
    model.train_predict_verification(glodbal_settings['network'])
    print('Printing the results...')
    model.plot_result_verification()
    

def hyper_parameter_phase(model):
    print('Split dataset 80/20 training/test...')
    model.data_split_80_20()
    print('Scaling process started...')
    model.scaling_data()
    print('Reshaped data for supervised learning...')
    model.series_to_supervised()
    print('Prepare data for test and verification...')
    model.prepare_verification_data()
       
    if glodbal_settings['network'] == 'rnn':  
        print('Building hyper parameter model Recurrent Neural Network...')
        print('Tuning hyper parameter model...')
        model.tuning_model_rnn()
        print('Printing the results...')
        model.plot_result_verification()
    
    if glodbal_settings['network'] == 'transformer':  
        print('Building hyper parameter model Transformer...')
        print('Tuning hyper parameter model...')
        model.tuning_model_transformer()
        print('Printing the results...')
        model.plot_result_verification()
    


    



    
