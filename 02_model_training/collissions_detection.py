# -*- coding: utf-8 -*-
# Import own modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CollissionModel:  
    def __init__(self, agents):
        self.agents = agents
        
        def scatter_plot_multiple_agents(self, agents):
            fig, ax = plt.subplots()
            markers = ['x', '.', 'o', '+']
            colors = ['g', 'b', 'r', 'c']
            data_x = []
            data_y = []
            x_min = []
            x_max = []
            y_min = []
            y_max = []
            
            for i in range(0, len(agents)):
                lng_file_x1 = 'saved_data/result_lng_' + agents[i] + '.csv'
                lat_file_y1 = 'saved_data/result_lat_' + agents[i] + '.csv'           
                x_1 = pd.read_csv(lng_file_x1)['Prediction lng'].values 
                y_1 = pd.read_csv(lat_file_y1)['Prediction lat'].values 
                data_x.append(x_1)
                data_y.append(y_1)        
                x_max.append(np.max(x_1))                
                y_max.append(np.max(y_1))
                x_min.append(np.min(x_1))          
                y_min.append(np.min(y_1))
            
            span = 0.0005
            this_x_min = min(x_min) - span
            this_x_max = max(x_min) + span
            this_y_min = min(y_min) - span
            this_y_max = max(y_max) + span
            title = 'Forcast movement: '
            legends = []
            for i in range(0, len(agents)):
                ax.scatter(data_x[i], data_y[i], c=colors[i], marker=markers[i])
                legends.append(agents[i])
                if i == len(agents)-1:
                    title = title + agents[i]
                else:
                    title = title + agents[i] + ' / '
            ax.set_xlim(this_x_min, this_x_max)
            ax.set_ylim(this_y_min, this_y_max)
            ax.ticklabel_format(useOffset=False)
            ax.set_title(title)
            ax.set_xlabel('longitude')
            ax.set_ylabel('latitude')
            plt.legend(legends)
            plt.show()
        
        
        def scatter_plot_collission_2D(self, agent_1, agent_2):
            lng_file_x1 = 'saved_data/result_lng_' + agent_1 + '.csv'
            lat_file_y1 = 'saved_data/result_lat_' + agent_1 + '.csv' 
            lng_file_x2 = 'saved_data/result_lng_' + agent_2 + '.csv'
            lat_file_y2 = 'saved_data/result_lat_' + agent_2 + '.csv'           
            x_1 = pd.read_csv(lng_file_x1)['Prediction lng'].values 
            y_1 = pd.read_csv(lat_file_y1)['Prediction lat'].values 
            x_2 = pd.read_csv(lng_file_x2)['Prediction lng'].values 
            y_2 = pd.read_csv(lat_file_y2)['Prediction lat'].values 
            
            self.detect_collissons_2D(x_1, x_2, y_1, y_2)
            col_x = self.collission_result_x
            col_y = self.collission_result_y
            sum_x = self._sum(col_x)
            sum_y = self._sum(col_y)
            
            this_x_min, this_y_min = self._min(x_1, x_2, y_1, y_2)
            this_x_max, this_y_max = self._max(x_1, x_2, y_1, y_2)
            
            fig, ax = plt.subplots()
            ax.scatter(x_1, y_1, c='g', marker='o')
            ax.scatter(x_2, y_2, c='b', marker='o')
            
            if sum_x == 0 and sum_y == 0:
                print('No collissions detected')
                col_legend = 'No collissions detected'
                ax.scatter(col_x, col_y, c='r', marker='x')
            else:
                print('Collissions detected')
                col_legend = 'Detected collissions'
                ax.scatter(col_x, col_y, c='r', marker='x')
            span = 0.0005
            ax.set_xlim(this_x_min - span, this_x_max + span)
            ax.set_ylim(this_y_min - span, this_y_max + span)
            ax.ticklabel_format(useOffset=False)
            ax.set_title('Forcast collissions: ' + agent_1 + ' / ' + agent_2)
            ax.set_xlabel('longitude')
            ax.set_ylabel('latitude')
            plt.legend([agent_1, agent_2, col_legend])
            plt.show()
            

        def detect_collissons_2D(self, x_1, x_2, y_1, y_2):
            collissions_x = []
            collissions_y = []
                        
            for y in range(0, len(x_1)):
                if abs(x_1[y] - x_2[y]) <= self.collission_span:
                    collissions_x.append((x_1[y] + x_2[y])/2)
                if abs(y_1[y] - y_2[y]) <= self.collission_span:
                    collissions_y.append((y_1[y] + y_2[y])/2)
                if abs(x_1[y] - x_2[y]) > self.collission_span:
                    collissions_x.append(0)
                if abs(y_1[y] - y_2[y]) > self.collission_span:
                    collissions_y.append(0)  

            for i in range(0, len(collissions_x)):
                if (collissions_x[i] and collissions_y[i]) != 0:
                    self.collission_result_x.append(collissions_x[i])    
                    self.collission_result_y.append(collissions_y[i]) 
                else:
                    self.collission_result_x.append(0)  
                    self.collission_result_y.append(0) 


        def _sum(self, arr):
            from functools import reduce
            sum = reduce(lambda a, b: a+b, arr)   
            return sum
        
        
        def _min(self, x_1, x_2, y_1, y_2):
            min_x_1 = np.min(x_1)
            min_x_2 = np.min(x_2)
            if min_x_1 < min_x_2:
                this_x_min = min_x_1
            else: 
                this_x_min = min_x_2
                
            min_y_1 = np.min(y_1)
            min_y_2 = np.min(y_2)
            if min_y_1 < min_y_2:
                this_y_min = min_y_1
            else:
                this_y_min = min_y_2
            return this_x_min, this_y_min
        
        
        def _max(self, x_1, x_2, y_1, y_2):
            max_x_1 = np.max(x_1)
            max_x_2 = np.max(x_2)
            if max_x_1 > max_x_2:
                this_x_max = max_x_1
            else: 
                this_x_max = max_x_2
                
            max_y_1 = np.max(y_1)
            max_y_2 = np.max(y_2)
            if max_y_1 > max_y_2:
                this_y_max = max_y_1
            else:
                this_y_max = max_y_2
            return this_x_max, this_y_max

