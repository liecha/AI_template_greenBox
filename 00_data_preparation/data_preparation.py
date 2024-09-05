# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import pandas as pd
from parameters import glodbal_settings

class DataPreparation:
    def __init__(self, agent, dataset):
        self.agent = agent
        self.dataset = dataset
        self.selected_data = []
            
        
    def select_agent(self):
        filter_dataset = self.dataset['bird_name'].values == self.agent
        selected_data = self.dataset.loc[filter_dataset] 
        selected_data.to_csv(glodbal_settings['data_path'])
        print('Selected agent: ', self.agent)
        print('Dataset for selected agent...') 
        print(selected_data.iloc[0])

    
    def plot_single_agent(self):   
        fig = plt.figure()
        fig.autofmt_xdate(rotation=45)       
        current_bird = pd.read_csv(glodbal_settings['data_path'])        
        title = "Flight path for bird " + self.agent
        current_bird.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)            
        plt.title(title)
      
        
    def save_data_to_csv(agent, mqtt):
        params = glodbal_settings['params']
        rows = []
        for i in range(0, len(params)):
            data_param = mqtt.agentscache.get_Train_Vector(agent, params[i])     
            rows.append(data_param)
        
        dict = {'lon': rows[0], 'lat': rows[1], 'alt': rows[2]}   
        df = pd.DataFrame(dict)
        df.to_csv(glodbal_settings['data_path'])