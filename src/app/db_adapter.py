import datetime

import mysql.connector


class DBAdapter:
    def __init__(self, config: dict[str, any]):
        # Database connection configuration
        self.config = config

        # SQL queries
        create_table_query = """
            CREATE TABLE IF NOT EXISTS datapoints (
                ts INT NOT NULL,
                temperature_celsius FLOAT NOT NULL,
                barometric_pressure FLOAT NOT NULL,
                humidity FLOAT NOT NULL,
                gas_resistance0 FLOAT NOT NULL,
                gas_resistance1 FLOAT NOT NULL,
                gas_resistance2 FLOAT NOT NULL,
                gas_resistance3 FLOAT NOT NULL,
                gas_resistance4 FLOAT NOT NULL,
                gas_resistance5 FLOAT NOT NULL,
                gas_resistance6 FLOAT NOT NULL,
                gas_resistance7 FLOAT NOT NULL,
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
                # print("MySQL connection closed.")

    def insert(self, table: str, datapoints: list[dict[str, any]]):
        try:
            # Establish the database connection
            connection = mysql.connector.connect(**self.config)
            cursor = connection.cursor()

            insert_query = f"""
                            INSERT INTO {table} (ts, 
                                                    temperature_celsius,    
                                                    barometric_pressure, 
                                                    humidity, 
                                                    gas_resistance0,
                                                    gas_resistance1,
                                                    gas_resistance2,
                                                    gas_resistance3,
                                                    gas_resistance4,
                                                    gas_resistance5,
                                                    gas_resistance6,
                                                    gas_resistance7, 
                                                    gps_latitude, 
                                                    gps_longitude, 
                                                    created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """

            # Insert each datapoint
            for datapoint in datapoints:
                values = (
                    datapoint["timestamp"],
                    datapoint["temperature_celsius"],
                    datapoint["barometric_pressure"],
                    datapoint["humidity"],
                    datapoint["gas_resistance0"],
                    datapoint["gas_resistance1"],
                    datapoint["gas_resistance2"],
                    datapoint["gas_resistance3"],
                    datapoint["gas_resistance4"],
                    datapoint["gas_resistance5"],
                    datapoint["gas_resistance6"],
                    datapoint["gas_resistance7"],
                    datapoint["gps_latitude"],
                    datapoint["gps_longitude"],
                    datetime.datetime.utcnow()
                )
                cursor.execute(insert_query, values)

                # Commit the transaction
                connection.commit()
                # print("Data inserted successfully!")
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                # print("MySQL connection closed.")


if __name__ == "__main__":
    db_config = {
        'user': 'ubuntu',
        'password': '2B-ornot-2B',
        'host': '54.166.148.213',
        'database': 'enose'
    }
    db_adapter = DBAdapter(db_config)
    db_adapter.insert(table="datapoints",
                      datapoints=[{
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
