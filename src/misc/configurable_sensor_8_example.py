import bme680

sensor = bme680.BME680()

# Configure the sensor
sensor.set_gas_heater_profile(1, duration=100)  # Select heater profile
sensor.set_gas_heater_temperature(300)
sensor.set_gas_heater_duration(150)
sensor.select_gas_heater_profile(0)

# Read sensor data
if sensor.get_sensor_data():
    gas_resistance = sensor.data.gas_resistance
    print(f"Gas resistance: {gas_resistance} ohms")
