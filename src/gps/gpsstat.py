import RPi.GPIO as GPIO
import time

# Configure GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)  # Example GPIO pin

def get_device_status(pin):
    """Returns the status of the GPIO pin."""
    return "High" if GPIO.input(pin) else "Low"

def main():
    try:
        while True:
            status = get_device_status(17)  # Check the status of GPIO 17
            print(f"GPIO 17 status: {status}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()
