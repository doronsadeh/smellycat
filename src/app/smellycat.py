import json
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from paho.mqtt import client as mqtt_client
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from app.geo_coloring import GeoColoring
from misc.random_hike import generate_random_hike


class SmellyCat:
    # Process only 1 in N messages to avoid piling up messages
    PROCESS_EVERY_1_IN_N_MESSAGES = 1
    NUM_SAMPLES_TO_PROCESS = 16

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

        self.previous_seg_color = []

        # Convert PCA results to a DataFrame for easier handling
        self.plot_df = pd.DataFrame(columns=['R', 'G', 'B'])

        start_latitude = 32.461692
        start_longitude = 34.962014
        self.hike_coordinates = generate_random_hike(start_latitude, start_longitude, total_distance_km=15, step_distance_m=5)
        self.step = 0
        self.gcol = GeoColoring()
        self.geomap = None

    class SensorData:
        def __init__(self, sensor_id, sensor_values):
            self.sensor_id = sensor_id
            self.timestamp = sensor_values[0]
            self.temperature = sensor_values[1]
            self.pressure = sensor_values[2] * 100.0
            self.humidity = sensor_values[3]
            self.gas_resistance = sensor_values[4]

    def oklab_to_srgb(self, L, a, b):
        """Convert Oklab values to sRGB values."""
        # Step 1: Convert Oklab to Linear RGB
        l = L + 0.3963377774 * a + 0.2158037573 * b
        m = L - 0.1055613458 * a - 0.0638541728 * b
        s = L - 0.0894841775 * a - 1.2914855480 * b

        l3 = l ** 3
        m3 = m ** 3
        s3 = s ** 3

        R_linear = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
        G_linear = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
        B_linear = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3

        # Step 2: Convert Linear RGB to sRGB
        def linear_to_srgb(x):
            return 12.92 * x if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055

        R = linear_to_srgb(R_linear)
        G = linear_to_srgb(G_linear)
        B = linear_to_srgb(B_linear)

        # Clamp values to [0, 1] and scale to [0, 255]
        R = np.clip(R, 0, 1) * 255
        G = np.clip(G, 0, 1) * 255
        B = np.clip(B, 0, 1) * 255

        return (int(R), int(G), int(B))

    def features_to_oklab_color(self, feature_vectors, smell):
        """
        Convert 8D feature vectors into Oklab-based RGB colors.

        Parameters:
            feature_vectors (numpy.ndarray): Array of shape (n_samples, 8), representing feature vectors.

        Returns:
            list: List of RGB tuples corresponding to each feature vector.
        """
        # Step 1: Normalize the feature vectors
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(feature_vectors)

        # Step 2: Dimensionality reduction to 3D using PCA
        pca = PCA(n_components=3)
        reduced_features = pca.fit_transform(normalized_features)

        # Step 3: Scale reduced features to valid Oklab ranges
        # Oklab ranges: L ∈ [0, 1], a and b roughly in [-1, 1] (scaled appropriately)
        oklab_scaler = MinMaxScaler(feature_range=(0, 1))
        reduced_features[:, 0] = oklab_scaler.fit_transform(reduced_features[:, [0]].reshape(-1, 1)).flatten()
        reduced_features[:, 1:] = reduced_features[:, 1:]  # Centered around 0 naturally by PCA

        # Calculate Z-scores
        df_rf = pd.DataFrame(reduced_features, columns=['L', 'a', 'b'])
        z_scores = np.abs(zscore(df_rf))

        # Define a threshold (e.g., Z-score > 3)
        threshold = np.percentile(z_scores, 99)
        outliers = (z_scores > threshold).any(axis=1)

        # Remove outliers
        df_cleaned = df_rf[~outliers]

        reduced_features = np.max(df_cleaned.to_numpy(), axis=0).reshape(1, -1)

        self.previous_seg_color.append(reduced_features)
        if len(self.previous_seg_color) > 3:
            self.previous_seg_color.pop(0)
            reduced_features = np.mean(self.previous_seg_color, axis=0).reshape(1, -1)

        # Step 4: Convert Oklab to sRGB
        rgb_colors = [self.oklab_to_srgb(L, a, b) for L, a, b in reduced_features]
        return rgb_colors, reduced_features

    def reducer(self, arr, smell, plot=False):
        rgb, reduced_features = self.features_to_oklab_color(arr, smell)

        if plot:
            self.plot_df = pd.concat([self.plot_df, pd.DataFrame(reduced_features, columns=['R', 'G', 'B'])], axis=0)

            # Create a 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot
            ax.scatter(
                self.plot_df['R'], self.plot_df['B'], self.plot_df['B'],
                c='blue', alpha=0.7, edgecolor='k'
            )

            # Add labels and a title
            ax.set_title("RGB Plot")
            ax.set_xlabel("R")
            ax.set_ylabel("G")
            ax.set_zlabel("B")

            # Show the plot
            plt.show()

        return rgb, reduced_features

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

    def feature_space_to_RGB(self, df, smell, plot=False):
        rgb, reduced_features = self.reducer(df[['gas_resistance0',
                                                 'gas_resistance1',
                                                 'gas_resistance2',
                                                 'gas_resistance3',
                                                 'gas_resistance4',
                                                 'gas_resistance5',
                                                 'gas_resistance6',
                                                 'gas_resistance7']],
                                             smell=smell,
                                             plot=plot)

        return rgb, reduced_features

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

        if len(self.sensor_data_df) >= max(3, self.NUM_SAMPLES_TO_PROCESS):
            rgb, reduced_features = self.feature_space_to_RGB(self.sensor_data_df.tail(max(3, self.NUM_SAMPLES_TO_PROCESS)), _smell, plot=False)

            c = self.hike_coordinates[self.step]
            r, g, b = rgb[0]
            self.geomap, bounding_box = self.gcol.create_colored_hexagon_map(self.geomap, c[1], c[0], resolution=13, color=f"rgba({r}, {g}, {b}, {0.8})")
            self.geomap.save("geomap.html")
            self.geomap = self.gcol.capture_and_crop_bounding_box(geomap=self.geomap,
                                                                  html_file=os.path.join(Path(__file__).parent, 'geomap.html'),
                                                                  bounding_box=bounding_box,
                                                                  output_image='smellycat_hike.png')
            self.geomap.save("geomap.html")

            self.step += 1

            try:
                response = requests.post('http://localhost:5000/update',
                                         data=json.dumps({"datapoint": [(_smell / 100000.0) * 100.0, list(reduced_features[0]), _temperature, _pressure, _humidity]}),
                                         headers={'Content-Type': 'application/json'})
                print(response.status_code, _smell, rgb, list(reduced_features[0]))
            except:
                pass

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
            if self.msg_count % self.PROCESS_EVERY_1_IN_N_MESSAGES == 0:
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
