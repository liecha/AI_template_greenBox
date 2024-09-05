# -*- coding: utf-8 -*-
# Import own modules
import sys
sys.path.append('../')
from main_functions import run_AI_waraps
import random
from connections import MQTTConnection
from mqtt_handler import MqttHandler

def connect_to_mqtt():
    random_int = random.randint(0, 1000)
    name = 'AI' + str(random_int)    
    connection: MQTTConnection = MQTTConnection()
    mqtt: MqttHandler = MqttHandler(connection, name)
    mqtt.connect()      
    return mqtt

mqtt = connect_to_mqtt()

run_AI_waraps(mqtt)