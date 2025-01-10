#!/usr/bin/env python3


#!/usr/bin/env python3
"""
GPS Coordinates to MQTT using GPIO
Author: Your Name
Date: 2025-01-10

Description:
This script reads GPS coordinates from a GPS module connected to GPIO pins
on a Raspberry Pi and sends the coordinates to a specified MQTT topic.

Usage:
    $ python3 gps_to_mqtt_gpio.py
    OR
    Make script executable and run:
    $ chmod +x gps_to_mqtt_gpio.py
    $ ./gps_to_mqtt_gpio.py
"""

import RPi.GPIO as GPIO
import time
import paho.mqtt.client as mqtt
import json

# Configuration for GPS and MQTT
GPS_PIN = 17  # GPIO pin connected to GPS TX line
MQTT_BROKER = '10.0.0.6'  # Change this to your broker's address
MQTT_PORT = 1883
MQTT_TOPIC = 'gps/coordinates'

# MQTT callbacks for API version 2
def on_connect(client, userdata, flags, rc, properties):
    print(f"Connected to MQTT broker with result code {rc}")

def on_publish(client, userdata, mid, reason_code, properties):
    print(f"Message {mid} published.")

# Setup MQTT client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)  # Explicitly use MQTT version 5
mqtt_client.on_connect = on_connect
mqtt_client.on_publish = on_publish

# Connect to MQTT broker
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(GPS_PIN, GPIO.IN)

def read_bit():
    """Reads a single bit from the GPS."""
    return GPIO.input(GPS_PIN)

def read_byte():
    """Reads a byte from the GPS using bit-banging."""
    byte = 0
    while read_bit():  # Wait for the start bit
        pass

    time.sleep(0.000104)  # Wait for the middle of the start bit

    for i in range(8):
        time.sleep(0.000104)  # Wait for bit period
        bit = read_bit()
        byte |= (bit << i)

    if byte != 0:
        publish_gps_data(byte, byte)

    time.sleep(0.000104)  # Wait for the stop bit
    return byte

def read_sentence():
    """Reads an NMEA sentence from the GPS."""
    sentence = bytearray()
    while True:
        byte = read_byte()
        if byte == ord('$'):  # Start of a new sentence
            sentence = bytearray()
        sentence.append(byte)
        if byte == ord('\n'):  # End of sentence
            return sentence.decode('ascii', errors='replace')

def parse_coordinates(gps_data):
    """Parses latitude and longitude from the NMEA sentence."""
    fields = gps_data.split(',')
    latitude = float(fields[2])  # Latitude in degrees
    longitude = float(fields[4])  # Longitude in degrees
    return latitude, longitude

def publish_gps_data(latitude, longitude):
    """Publishes the GPS coordinates to the MQTT topic."""
    message = {
        'latitude': latitude,
        'longitude': longitude,
        'timestamp': time.time()
    }
    mqtt_client.publish(MQTT_TOPIC, json.dumps(message))

def main():
    try:
        while True:
            gps_data = read_sentence()
            latitude, longitude = parse_coordinates(gps_data)
            publish_gps_data(latitude, longitude)
            time.sleep(5)  # Delay between GPS readings
    except KeyboardInterrupt:
        print("Stopping GPS to MQTT...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    mqtt_client.loop_start()  # Start MQTT loop to handle network traffic
    main()
