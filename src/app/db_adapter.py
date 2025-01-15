import datetime

import mysql.connector


class DBAdapter:
    def __init__(self):
        # Database connection configuration
        self.config = {
            'user': 'ubuntu',
            'password': '2B-ornot-2B',
            'host': '54.166.148.213',
            'database': 'enose'
        }

        # SQL queries
        create_table_query = """
            CREATE TABLE IF NOT EXISTS datapoints (
                sensor_id INT NOT NULL,
                timestamp INT NOT NULL,
                temperature_celsius FLOAT NOT NULL,
                barometric_pressure FLOAT NOT NULL,
                humidity FLOAT NOT NULL,
                gas_resistance FLOAT NOT NULL,
                gps_latitude FLOAT NOT NULL,
                gps_longitude FLOAT NOT NULL,
                created_at DATETIME NOT NULL
            );
            """

        try:
            # Establish the database connection
            connection = mysql.connector.connect(**self.config)
            cursor = connection.cursor()

            # Check and create table if not exists
            cursor.execute(create_table_query)
            connection.commit()
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection closed.")

    def insert(self, datapoints: list[dict[str, any]]):
        try:
            # Establish the database connection
            connection = mysql.connector.connect(**self.config)
            cursor = connection.cursor()

            insert_query = """
                            INSERT INTO datapoints (sensor_id, 
                                                    timestamp, 
                                                    temperature_celsius,    
                                                    barometric_pressure, 
                                                    humidity, 
                                                    gas_resistance, 
                                                    gps_latitude, 
                                                    gps_longitude, 
                                                    created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """

            # Insert each datapoint
            for datapoint in datapoints:
                values = (
                    datapoint["sensor_id"],
                    datapoint["timestamp"],
                    datapoint["temperature_celsius"],
                    datapoint["barometric_pressure"],
                    datapoint["humidity"],
                    datapoint["gas_resistance"],
                    datapoint["gps_latitude"],
                    datapoint["gps_longitude"],
                    datetime.datetime.now()
                )
                cursor.execute(insert_query, values)

                # Commit the transaction
                connection.commit()
                print("Data inserted successfully!")
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection closed.")


if __name__ == "__main__":
    db_adapter = DBAdapter()
    db_adapter.insert([
        {
            "sensor_id": 1,
            "timestamp": 1677721600,
            "temperature_celsius": 25.0,
            "barometric_pressure": 1013.25,
            "humidity": 50.0,
            "gas_resistance": 1000.0,
            "gps_latitude": 40.7128,
            "gps_longitude": -74.0060,
        }
    ])