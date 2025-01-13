import RPi.GPIO as GPIO
import time

# Setup GPIO
GPIO.setmode(GPIO.BCM)  # Use BCM numbering
TX_PIN = 17  # GPIO pin to send data (use any available pin)

_50ms = 0.001

GPIO.setup(TX_PIN, GPIO.OUT)

# Simulated GPS data (NMEA sentence format)
gps_data = [
    "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47",
    "$GPGLL,4916.45,N,12311.12,W,225444,A,*1D",
    "$GPGSA,A,3,04,05,..,29,34,,,1.8,1.0,1.5*33",
]


def send_bit(pin, bit):
    """Send a single bit over GPIO."""
    GPIO.output(pin, bit)
    time.sleep(_50ms * 2.0)


def send_byte(pin, byte):
    """Send a byte over GPIO by sending each bit."""
    for i in range(8):
        bit = (byte >> i) & 1
        send_bit(pin, bit)


def send_gps_data(pin, data):
    """Send a string of GPS data over GPIO."""
    for char in data:
        send_byte(pin, ord(char))
    # Send a newline character to end the transmission
    send_byte(pin, ord('\n'))


try:
    while True:
        for sentence in gps_data:
            send_gps_data(TX_PIN, sentence)
            print(sentence)
            time.sleep(1)  # Delay between sentences
finally:
    print('GPS sent')
    GPIO.cleanup()
