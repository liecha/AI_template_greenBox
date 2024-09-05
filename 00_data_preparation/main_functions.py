# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
import pandas as pd

# Import own modules
from data_preparation import DataPreparation

def read_data_from_file(agent):    
    dataset = pd.read_csv('../saved_data/bird_migration.csv') 
    prepModule = DataPreparation(agent, dataset)
    prepModule.select_agent()
    prepModule.plot_single_agent()