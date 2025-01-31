#!/bin/sh
(
echo "Waiting for Wi-Fi connection..."
while ! ping -c 1 -W 1 8.8.8.8 &> /dev/null; do
    echo "Wi-Fi not connected. Retrying..."
    sleep 1
done
echo "Wi-Fi is connected."

echo "Running start GPS ..."

export BROKER_PASSWORD=2B-ornot-2B

echo "Starting virtualenv ..."

cd /home/pi
source "./gpsenv/bin/activate"

echo "Running GPS listener ..."

cd /home/pi/enose/src/gpsneo6
python -u ./neo6_parser.py
) 2>&1 | tee -a gps.log
