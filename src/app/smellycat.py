import json
import random

import numpy as np
import requests
# import requests
from paho.mqtt import client as mqtt_client


class SmellyCat:
    # REST_URL = "http://127.0.0.1:7000/v1/odour-detection"
    # HEADERS = {"Content-Type": "application/json"}

    # Process only 1 in N messages to avoid piling up messages
    RPOCESS_EVERY_1_IN_N_MESSAGES = 1

    msg_count = 0

    def __init__(self, broker, port, topic, client_id, username, password):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.username = username
        self.password = password

    class SensorData:
        def __init__(self, sensor_id, sensor_values):
            self.sensor_id = sensor_id
            self.timestamp = sensor_values[0]
            self.temperature = sensor_values[1]
            self.pressure = sensor_values[2] * 100.0
            self.humidity = sensor_values[3]
            self.gas_resistance = sensor_values[4]

    def smell(self, sensor_data: str):
        sensors = []
        for _sdata in eval(sensor_data)['datapoints']:
            sensors.append(self.SensorData(_sdata[0], _sdata[1:]))

        _smell = np.mean([s.gas_resistance for s in sensors])
        _pressure = np.mean([s.pressure for s in sensors])
        _humidity = np.mean([s.humidity for s in sensors])
        _temprature = np.mean([s.temperature for s in sensors])

        return _smell, _temprature, _pressure, _humidity

    def process_sensor_data(self, sensor_data):
        _smell, _temperature, _pressure, _humidity = self.smell(sensor_data)
        response = requests.post('http://localhost:5000/update',
                                 data=json.dumps({"datapoint": [(_smell / 100000.0) * 100.0, _temperature, _pressure, _humidity]}),
                                 headers={'Content-Type': 'application/json'})
        # print(response.json())
        print(_smell)

    def connect_mqtt(self):
        def on_connect(self, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker: {}".format(broker))
            else:
                print("Failed to connect, erorr {}}".format(rc))

        client = mqtt_client.Client(self.client_id)
        client.username_pw_set(self.username, self.password)
        client.on_connect = on_connect
        client.connect(broker, port)
        return client

    def subscribe(self, client, topic):
        def on_message(client, userdata, msg):
            self.msg_count += 1
            if self.msg_count % self.RPOCESS_EVERY_1_IN_N_MESSAGES == 0:
                self.process_sensor_data(msg.payload.decode())
                # print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

        client.subscribe(topic)
        client.on_message = on_message

    def run(self):
        client = self.connect_mqtt()
        self.subscribe(client, topic)
        client.loop_forever()


if __name__ == "__main__":
    ###########################################################
    # MQTT Broker info
    ###########################################################
    broker = '10.0.0.6'
    port = 1883
    topic = "sensorData"
    client_id = f'eNose-{random.randint(0, 100)}'
    username = 'admin'
    password = 'admin'

    r = SmellyCat(broker, port, topic, client_id, username, password)
    r.run()
