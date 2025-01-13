#!/usr/bin/env python3

import time

from gps3 import gps3


def main():
    # Create gps socket and data stream
    gps_socket = gps3.GPSDSocket()
    data_stream = gps3.DataStream()

    # Connect to the local gpsd
    gps_socket.connect()
    # Start watching gps data
    gps_socket.watch()

    try:
        while True:
            # gps_socket is an iterator that yields NMEA/JSON data from gpsd
            for new_data in gps_socket:
                if new_data:
                    # Unpack the incoming data into a dictionary
                    data_stream.unpack(new_data)

                    # data_stream.TPV contains Time-Position-Velocity data
                    # Check the dictionary keys and handle missing data carefully
                    latitude = data_stream.TPV['lat']
                    longitude = data_stream.TPV['lon']
                    altitude = data_stream.TPV['alt']
                    speed = data_stream.TPV['speed']
                    track = data_stream.TPV['track']
                    time_utc = data_stream.TPV['time']

                    # GPS status
                    mode = data_stream.TPV['mode']  # 0 = no fix, 1 = no fix, 2 = 2D fix, 3 = 3D fix

                    # Print GPS info (when available)
                    # Note: Some fields might be 'n/a' or None if no fix yet.
                    print(f"Time (UTC): {time_utc}")
                    print(f"Latitude: {latitude}")
                    print(f"Longitude: {longitude}")
                    print(f"Altitude: {altitude}")
                    print(f"Speed (m/s): {speed}")
                    print(f"Track (deg): {track}")
                    print(f"Mode: {mode}")
                    print("-----------------------------------")

                # Sleep briefly to avoid flooding your terminal
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("User interrupted.")
    finally:
        # Optionally stop watching GPS data
        gps_socket.close()
        print("GPS reading stopped.")


if __name__ == '__main__':
    main()
