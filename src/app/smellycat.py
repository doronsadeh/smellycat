import json
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import rgb2hex
from paho.mqtt import client as mqtt_client
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class SmellyCat:
    # Process only 1 in N messages to avoid piling up messages
    RPOCESS_EVERY_1_IN_N_MESSAGES = 3

    msg_count = 0

    def __init__(self, broker, port, topic, client_id, username, password):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.username = username
        self.password = password
        self.sensor_data_df = pd.DataFrame(columns=[
            'timestamp',
            'temperature',
            'pressure',
            'humidity',
            'gas_resistance0',
            'gas_resistance1',
            'gas_resistance2',
            'gas_resistance3',
            'gas_resistance4',
            'gas_resistance5',
            'gas_resistance6',
            'gas_resistance7',
        ])

    class SensorData:
        def __init__(self, sensor_id, sensor_values):
            self.sensor_id = sensor_id
            self.timestamp = sensor_values[0]
            self.temperature = sensor_values[1]
            self.pressure = sensor_values[2] * 100.0
            self.humidity = sensor_values[3]
            self.gas_resistance = sensor_values[4]

    def reducer(self, df, plot=True):
        # Perform PCA to reduce to 2 dimensions
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(df)

        # Create a DataFrame for the PCA results
        pca_df = pd.DataFrame(pca_result, columns=['R', 'G', 'B'])

        if plot:
            # Plot the 2D PCA results
            plt.figure(figsize=(8, 6))
            plt.scatter(pca_df['R'], pca_df['G'], pca_df['B'], alpha=0.7, c='blue', edgecolor='k')
            plt.title('RGB Space of Sensor Data (PCA Reduced to 3 Dimensions)')
            plt.xlabel('R')
            plt.ylabel('G')
            plt.ylabel('B')
            plt.grid(True)
            plt.show()

            # Print explained variance ratio
            print("Explained Variance Ratio:", pca.explained_variance_ratio_)

        return pca_df.to_dict(orient='records'), pca.explained_variance_ratio_

    def biased_average(self, data, p):
        """
        Calculate a biased average of a list of numbers.

        Parameters:
            data (list): A list of numerical values.
            p (float): The bias factor. p > 1 biases towards larger values,
                       0 < p < 1 reduces the bias, and p < 0 biases towards smaller values.

        Returns:
            float: The biased average.
        """
        if not data or p == 0:
            raise ValueError("Data must be a non-empty list and p must not be 0.")

        n = len(data)
        return (sum(x ** p for x in data) / n) ** (1 / p)

    def smell(self, sensor_data: str):
        sensors = []
        for _sdata in eval(sensor_data)['datapoints']:
            sensors.append(self.SensorData(_sdata[0], _sdata[1:]))

        _smell = self.biased_average([s.gas_resistance for s in sensors], p=1.75)
        _pressure = np.mean([s.pressure for s in sensors])
        _humidity = np.mean([s.humidity for s in sensors])
        _temperature = np.mean([s.temperature for s in sensors])

        return _smell, _temperature, _pressure, _humidity

    def feature_space_to_RGB(self, df):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(df[['temperature',
                                                   'pressure',
                                                   'humidity',
                                                   'gas_resistance0',
                                                   'gas_resistance1',
                                                   'gas_resistance2',
                                                   'gas_resistance3',
                                                   'gas_resistance4',
                                                   'gas_resistance5',
                                                   'gas_resistance6',
                                                   'gas_resistance7'
                                                   ]])

        rgb_dict, var_ratio = self.reducer(normalized_data, plot=False)

        rgb_df = pd.DataFrame(rgb_dict)
        scaler = MinMaxScaler()
        scaled_RGB = np.mean(scaler.fit_transform(rgb_df)*255.0, axis=1).astype(int)

        return scaled_RGB, var_ratio

    def process_sensor_data(self, sensor_data):
        _smell, _temperature, _pressure, _humidity = self.smell(sensor_data)

        # Store time series
        dpoints = [s[5] for s in eval(sensor_data)['datapoints']]
        sample_df = pd.DataFrame({
            'timestamp': datetime.now(),
            'temperature': _temperature,
            'pressure': _pressure,
            'humidity': _humidity,
            'gas_resistance0': dpoints[0],
            'gas_resistance1': dpoints[1],
            'gas_resistance2': dpoints[2],
            'gas_resistance3': dpoints[3],
            'gas_resistance4': dpoints[4],
            'gas_resistance5': dpoints[5],
            'gas_resistance6': dpoints[6],
            'gas_resistance7': dpoints[7],
        }, index=[0])

        self.sensor_data_df = pd.concat([self.sensor_data_df, sample_df], ignore_index=True)

        if len(self.sensor_data_df) >= 3:
            rgb_dict, var_ratio = self.feature_space_to_RGB(self.sensor_data_df.tail(3))

        if len(self.sensor_data_df) > 100:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(self.sensor_data_df[['temperature',
                                                                        'pressure',
                                                                        'humidity',
                                                                        'gas_resistance0',
                                                                        'gas_resistance1',
                                                                        'gas_resistance2',
                                                                        'gas_resistance3',
                                                                        'gas_resistance4',
                                                                        'gas_resistance5',
                                                                        'gas_resistance6',
                                                                        'gas_resistance7'
                                                                        ]])

            self.reducer(normalized_data)

            self.sensor_data_df = pd.DataFrame(columns=[
                'timestamp',
                'temperature',
                'pressure',
                'humidity',
                'gas_resistance0',
                'gas_resistance1',
                'gas_resistance2',
                'gas_resistance3',
                'gas_resistance4',
                'gas_resistance5',
                'gas_resistance6',
                'gas_resistance7',
            ])

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
