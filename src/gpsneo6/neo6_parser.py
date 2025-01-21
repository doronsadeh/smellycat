import serial
import pynmea2

# GPS device settings
port = "/dev/ttyACM0"  # Update with your GPS device port
baud_rate = 9600       # Default baud rate for Neo-6 GPS

def parse_gps_data():
    try:
        with serial.Serial(port, baud_rate, timeout=1) as gps_serial:
            print("Listening to GPS data...")
            while True:
                line = gps_serial.readline().decode('ascii', errors='ignore')  # Read raw data
                if line.startswith('$GPGGA'):  # Global Positioning System Fix Data
                    gpgga = pynmea2.parse(line)
                    print(f"Time (UTC): {gpgga.timestamp}")
                    print(f"Latitude: {gpgga.latitude} {gpgga.lat_dir}")
                    print(f"Longitude: {gpgga.longitude} {gpgga.lon_dir}")
                    print(f"Fix Quality: {gpgga.gps_qual} (1 = No Fix, 2 = DGPS Fix)")
                    print(f"Number of Satellites: {gpgga.num_sats}")
                    print(f"Altitude: {gpgga.altitude} {gpgga.altitude_units}")
                    print(f"GeoID Separation: {gpgga.geo_sep} {gpgga.geo_sep_units}")
                    print("-" * 50)

                elif line.startswith('$GPRMC'):  # Recommended Minimum Specific GNSS Data
                    gprmc = pynmea2.parse(line)
                    print(f"Time (UTC): {gprmc.timestamp}")
                    print(f"Latitude: {gprmc.latitude} {gprmc.lat_dir}")
                    print(f"Longitude: {gprmc.longitude} {gprmc.lon_dir}")
                    print(f"Speed (knots): {gprmc.spd_over_grnd}")
                    print(f"Course: {gprmc.true_course}°")
                    print("-" * 50)

            # TODO send data to MQTT gps/location

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except pynmea2.ParseError as e:
        print(f"NMEA parse error: {e}")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the GPS parser
if __name__ == "__main__":
    broker_config = {
        "broker": "54.166.148.213",
        "port": 1883,
        "sensor_topic": "sensorData",
        "gps_topic": "gps/location",
        "username": "ubuntu",
        "password": "2B-ornot-2B",
    }

    parse_gps_data()
