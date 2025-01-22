import json
import random
from datetime import datetime
from time import sleep

import pynmea2
import serial
from paho.mqtt import client as mqtt_client
from paho.mqtt.client import MQTTv5
from paho.mqtt.enums import CallbackAPIVersion


class GPSNeo6Parser:
    # GPS device settings
    serial_port = "/dev/ttyACM0"  # Update with your GPS device port
    baud_rate = 9600  # Default baud rate for Neo-6 GPS

    def __init__(self, broker_config):
        self.client = None
        self.broker = broker_config['broker']
        self.port = broker_config['port']
        self.gps_topic = broker_config['gps_topic']
        self.username = broker_config['username']
        self.password = broker_config['password']

    def to_epoch(self, timestamp):
        return int(timestamp.timestamp())

    def parse_gps_data(self):
        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=1) as gps_serial:
                print("Listening to GPS data...")
                while True:
                    gps_data = None
                    line = gps_serial.readline().decode('ascii', errors='ignore')  # Read raw data
                    if line.startswith('$GPGGA'):  # Global Positioning System Fix Data
                        gpgga = pynmea2.parse(line)
                        gps_data = {
                            "timestamp": self.to_epoch(datetime.utcnow()),
                            "latitude": gpgga.latitude,
                            "latitude_direction": gpgga.lat_dir,
                            "longitude": gpgga.longitude,
                            "longitude_direction": gpgga.lon_dir,
                            "fix_quality": gpgga.gps_qual,  # 1 = No Fix, 2 = DGPS Fix
                            "num_sats": gpgga.num_sats,
                            "altitude": gpgga.altitude,
                            "altitude_units": gpgga.altitude_units,
                            "geo_sep": gpgga.geo_sep,
                            "geo_sep_units": gpgga.geo_sep_units
                        }

                    elif line.startswith('$GPRMC'):  # Recommended Minimum Specific GNSS Data
                        gprmc = pynmea2.parse(line)
                        gps_data = {
                            "timestamp": self.to_epoch(datetime.utcnow()),
                            "latitude": gprmc.latitude,
                            "latitude_direction": gprmc.lat_dir,
                            "longitude": gprmc.longitude,
                            "longitude_direction": gprmc.lon_dir,
                            "speed_knots": gprmc.spd_over_grnd,
                            "true_course": gprmc.true_course
                        }

                    if gps_data is not None:
                        # Send data to MQTT gps/location
                        print(gps_data)
                        payload = json.dumps(gps_data)  # Convert to JSON string
                        self.client.publish(self.gps_topic, payload)
                        print(f"Published: {payload} to topic: {self.gps_topic}")

                    sleep(1)

        except serial.SerialException as e:
            print(f"Serial error: {e}")
        except pynmea2.ParseError as e:
            print(f"NMEA parse error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"An error occurred: {e}")

    def connect_mqtt(self):
        def on_connect(client, userdata, flags, reasonCode, properties):
            if reasonCode == 0:
                client.subscribe(self.gps_topic)
                print("Connected to MQTT Broker: {}".format(client.extra_data["broker"]))
            else:
                print("Failed to connect, erorr {}}".format(reasonCode))

        def on_message(client, userdata, msg):
            print(f"Received from `{msg.topic}` topic: `{msg.payload.decode()}`")

        def on_publish(client, userdata, mid):
            print(f"Message {mid} published")

        def on_log(client, userdata, level, buf):
            print("Log: ", buf)

        def on_disconnect(client, userdata, rc, properties=None):
            if rc != 0:
                print("Unexpected disconnection.")

        client = mqtt_client.Client(client_id=f"eNose-{random.randint(0, 1000)}",
                                    callback_api_version=CallbackAPIVersion.VERSION2,
                                    protocol=MQTTv5)
        client.username_pw_set(self.username, self.password)
        client.extra_data = {"broker": self.broker}
        client.on_connect = on_connect
        client.on_message = on_message
        client.on_log = on_log
        client.on_disconnect = on_disconnect
        client.connect(self.broker, self.port, clean_start=True, keepalive=60)
        return client

    def run(self):
        self.client = self.connect_mqtt()
        self.parse_gps_data()


# Run the GPS parser
if __name__ == "__main__":
    broker_config = {
        "broker": "54.166.148.213",
        "port": 1883,
        "gps_topic": "gps/location",
        "username": "ubuntu",
        "password": "2B-ornot-2B",
    }

    g = GPSNeo6Parser(broker_config)
    g.run()
