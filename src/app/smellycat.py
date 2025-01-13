import random

# import requests
from paho.mqtt import client as mqtt_client


class SmellyCat:
    # REST_URL = "http://127.0.0.1:7000/v1/odour-detection"
    # HEADERS = {"Content-Type": "application/json"}

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

    def process_sensor_data(self, sensor_data):
        # response = requests.post(REST_URL, data=sensor_data, headers=HEADERS)
        # print(response.json())
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
            if self.msg_count % self.RPOCESS_EVERY_1_IN_N_MESSAGES == 0:
                self.process_sensor_data(msg.payload.decode())
                print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

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
