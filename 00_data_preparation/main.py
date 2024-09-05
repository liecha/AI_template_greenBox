# -*- coding: utf-8 -*-
# Import own modules
from main_functions import read_data_from_file
from parameters import glodbal_settings

agents = glodbal_settings['agents']

read_data_from_file(agents[0])