/**
 * Copyright (C) 2021 Bosch Sensortec GmbH
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */

/* The new sensor needs to be conditioned before the example can work reliably. You may run this
 * example for 24hrs to let the sensor stabilize.
 */

/**
* This sketch was derived from the basic.ino sketch described below.
*
 * basic.ino sketch :
 * This is an example for illustrating the BSEC virtual outputs using BME688 Development Kit,
 * which has been designed to work with Adafruit ESP32 Feather Board
 * For more information visit : 
 * https://www.bosch-sensortec.com/software-tools/software/bme688-software/
 */

#include <bsec2.h>
#include "commMux.h"

#include "mqtt_datalogger.h"
#include <WiFi.h>
#include <PubSubClient.h>

/* Macros used */
/* Number of sensors to operate*/
#define NUM_OF_SENS    8
#define PANIC_LED   LED_BUILTIN
#define ERROR_DUR   1000

// WiFi parameters
const char* ssid = "Smellycat";
const char* password = "0544502042";

// MQTT parameters
const char* mqttServer = "54.166.148.213";
const int mqttPort = 1883;
const char* mqttUser = "ubuntu";
const char* mqttPassword = "2B-ornot-2B";
const char* mqttTopic = "sensorData";
const char* mqttClientName = "BME688";

// create the clients
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// create MQTT logger
bme68xData sensorData[NUM_OF_SENS] = {0};
mqttDataLogger logger(&mqttClient, NUM_OF_SENS, mqttTopic);

void reconnect() {
  if (WiFi.status() != WL_CONNECTED) {
    // First try to reconnect to WiFi if needed
    Serial.println("Attempting to connect to WiFi ...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
          Serial.print(WiFi.status());
          delay(500);
          Serial.print(".");
    }

    Serial.println("\nReconnected to WiFi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Reconnect to NTP and re-sync time
    synchroniseWith_NTP_Time();
  }

  // Loop until we're reconnected to MQTT
  while (!mqttClient.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Attempt to connect
    if (mqttClient.connect(mqttClientName, mqttUser, mqttPassword )) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(mqttClient.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}
/* Helper functions declarations */
/**
 * @brief : This function toggles the led when a fault was detected
 */
void errLeds(void);

/**
 * @brief : This function checks the BSEC status, prints the respective error code. Halts in case of error
 * @param[in] bsec  : Bsec2 class object
 */
void checkBsecStatus(Bsec2 bsec);

/**
 * @brief : This function is called by the BSEC library when a new output is available
 * @param[in] input     : BME68X sensor data before processing
 * @param[in] outputs   : Processed BSEC BSEC output data
 * @param[in] bsec      : Instance of BSEC2 calling the callback
 */
void newDataCallback(const bme68xData data, const bsecOutputs outputs, Bsec2 bsec);

/* Create an array of objects of the class Bsec2 */
Bsec2 envSensor[NUM_OF_SENS];
commMux communicationSetup[NUM_OF_SENS];
uint8_t bsecMemBlock[NUM_OF_SENS][BSEC_INSTANCE_SIZE];
uint8_t sensor = 0;

const char* ntpServer = "0.pool.ntp.org";
const long  gmtOffset_sec = 2 * 60 * 60;
const int   daylightOffset_sec = 0;

#include <time.h>                   // time() ctime()
time_t now;                         // this is the epoch
tm myTimeInfo;                      // the structure tm holds time information in a more convient way

void showTime() {
  time(&now);                       // read the current time
  localtime_r(&now, &myTimeInfo);           // update the structure tm with the current time
  Serial.print("year:");
  Serial.print(myTimeInfo.tm_year + 1900);  // years since 1900
  Serial.print("\tmonth:");
  Serial.print(myTimeInfo.tm_mon + 1);      // January = 0 (!)
  Serial.print("\tday:");
  Serial.print(myTimeInfo.tm_mday);         // day of month
  Serial.print("\thour:");
  Serial.print(myTimeInfo.tm_hour);         // hours since midnight  0-23
  Serial.print("\tmin:");
  Serial.print(myTimeInfo.tm_min);          // minutes after the hour  0-59
  Serial.print("\tsec:");
  Serial.print(myTimeInfo.tm_sec);          // seconds after the minute  0-61*
  Serial.print("\twday");
  Serial.print(myTimeInfo.tm_wday);         // days since Sunday 0-6
  if (myTimeInfo.tm_isdst == 1)             // Daylight Saving Time flag
    Serial.print("\tDST");
  else
    Serial.print("\tstandard");

  Serial.println();
}

void synchroniseWith_NTP_Time() {
  Serial.print("ConfigTime uses ntpServer ");
  Serial.println(ntpServer);
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  Serial.print("synchronising time");

  while (myTimeInfo.tm_year + 1900 < 2000 ) {
    time(&now);                       // read the current time
    localtime_r(&now, &myTimeInfo);
    delay(100);
    Serial.print(".");
  }
  Serial.print("\n time synchronsized \n");
  showTime();
}

/* Entry point for the example */
void setup(void)
{
    /* Desired subscription list of BSEC2 outputs */
    bsecSensor sensorList[] = {
            BSEC_OUTPUT_IAQ,
            BSEC_OUTPUT_STATIC_IAQ,
            BSEC_OUTPUT_CO2_EQUIVALENT,
            BSEC_OUTPUT_BREATH_VOC_EQUIVALENT,
            BSEC_OUTPUT_RAW_TEMPERATURE,
            BSEC_OUTPUT_RAW_PRESSURE,
            BSEC_OUTPUT_RAW_HUMIDITY,
            BSEC_OUTPUT_RAW_GAS,
            BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_TEMPERATURE,
            BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_HUMIDITY,
            BSEC_OUTPUT_GAS_PERCENTAGE,
            BSEC_OUTPUT_STABILIZATION_STATUS,
            BSEC_OUTPUT_RUN_IN_STATUS,
    };

    /* Initialize the communication interfaces */
    Serial.begin(115200);
    commMuxBegin(Wire, SPI);
    pinMode(PANIC_LED, OUTPUT);
    delay(100);
    /* Valid for boards with USB-COM. Wait until the port is open */
    while(!Serial) delay(10);

    mqttClient.setServer(mqttServer, mqttPort);
    mqttClient.setBufferSize(600);
    Serial.print("MQTT client buffer size: ");
    Serial.println(mqttClient.getBufferSize());

    reconnect();

    logger.beginSensorData();

    for (uint8_t i = 0; i < NUM_OF_SENS; i++)
    {
        /* Sets the Communication interface for the sensors */
        communicationSetup[i] = commMuxSetConfig(Wire, SPI, i, communicationSetup[i]);

        /* Assigning a chunk of memory block to the bsecInstance */
         envSensor[i].allocateMemory(bsecMemBlock[i]);

        /* Initialize the library and interfaces */
        if (!envSensor[i].begin(BME68X_SPI_INTF, commMuxRead, commMuxWrite, commMuxDelay, &communicationSetup[i]))
        {
            checkBsecStatus (envSensor[i]);
        }

        /* Subscribe to the desired BSEC2 outputs */
        if (!envSensor[i].updateSubscription(sensorList, ARRAY_LEN(sensorList), BSEC_SAMPLE_RATE_LP))
        {
            checkBsecStatus (envSensor[i]);
        }

        /* Whenever new data is available call the newDataCallback function */
        envSensor[i].attachCallback(newDataCallback);


    }

    Serial.println("BSEC library version " + \
            String(envSensor[0].version.major) + "." \
            + String(envSensor[0].version.minor) + "." \
            + String(envSensor[0].version.major_bugfix) + "." \
            + String(envSensor[0].version.minor_bugfix));
}

/* Function that is looped forever */
void loop(void)
{
    /* Call the run function often so that the library can
     * check if it is time to read new data from the sensor
     * and process it.
     */
    for (sensor = 0; sensor < NUM_OF_SENS; sensor++)
    {
        if (!envSensor[sensor].run())
        {
         checkBsecStatus(envSensor[sensor]);
        }
    }
    reconnect();
}

void errLeds(void)
{
    while(1)
    {
        digitalWrite(PANIC_LED, HIGH);
        delay(ERROR_DUR);
        digitalWrite(PANIC_LED, LOW);
        delay(ERROR_DUR);
    }
}

void newDataCallback(const bme68xData data, const bsecOutputs outputs, Bsec2 bsec)
{
    if (!outputs.nOutputs)
    {
        return;
    }

    //Serial.println("BSEC outputs:\n\tsensor num = " + String(sensor));
    //Serial.println("\ttimestamp = " + String((int) (outputs.output[0].time_stamp / INT64_C(1000000))));
    for (uint8_t i = 0; i < outputs.nOutputs; i++)
    {
        const bsecData output  = outputs.output[i];
        switch (output.sensor_id)
        {
            case BSEC_OUTPUT_IAQ:
                //Serial.println("\tiaq = " + String(output.signal));
                //Serial.println("\tiaq accuracy = " + String((int) output.accuracy));
                break;
            case BSEC_OUTPUT_STATIC_IAQ:
                //Serial.println("\tstatic iaq = " + String(output.signal));
                //Serial.println("\tiaq accuracy = " + String((int) output.accuracy));
                break;
            case BSEC_OUTPUT_CO2_EQUIVALENT:
                //Serial.println("\tCO2 equiv. = " + String(output.signal));
                break;
            case BSEC_OUTPUT_BREATH_VOC_EQUIVALENT:
                //Serial.println("\tBreath VOC = " + String(output.signal));
                break;
            case BSEC_OUTPUT_RAW_TEMPERATURE:
                //Serial.println("\ttemperature = " + String(output.signal));
                break;
            case BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_TEMPERATURE:
                //Serial.println("\tTemprature (compensated) = " + String(output.signal));
                sensorData[sensor].temperature = output.signal;
                break;
            case BSEC_OUTPUT_RAW_PRESSURE:
                //Serial.println("\tpressure = " + String(output.signal));
                sensorData[sensor].pressure = output.signal;
                break;
            case BSEC_OUTPUT_RAW_HUMIDITY:
                //Serial.println("\thumidity = " + String(output.signal));
                break;
            case BSEC_OUTPUT_SENSOR_HEAT_COMPENSATED_HUMIDITY:
                //Serial.println("\tHumidity (compensated) = " + String(output.signal));
                sensorData[sensor].humidity = output.signal;
                break;
            case BSEC_OUTPUT_RAW_GAS:
                //Serial.println("\tgas resistance = " + String(output.signal));
                sensorData[sensor].gas_resistance = output.signal;
                break;
            case BSEC_OUTPUT_GAS_PERCENTAGE:
                //Serial.println("\tgas percent. = " + String(output.signal));
                break;
            case BSEC_OUTPUT_STABILIZATION_STATUS:
                //Serial.println("\tstabilization status = " + String(output.signal));
                break;
            case BSEC_OUTPUT_RUN_IN_STATUS:
                //Serial.println("\trun in status = " + String(output.signal));
                break;
            default:
                break;
        }
    }
     logger.assembleAndPublishSensorData(sensor, &sensorData[sensor]);
}

void checkBsecStatus(Bsec2 bsec)
{
    if (bsec.status < BSEC_OK)
    {
        Serial.println("BSEC error code : " + String(bsec.status));
        errLeds(); /* Halt in case of failure */
    }
    else if (bsec.status > BSEC_OK)
    {
        Serial.println("BSEC warning code : " + String(bsec.status));
    }

    if (bsec.sensor.status < BME68X_OK)
    {
        Serial.println("BME68X error code : " + String(bsec.sensor.status));
        errLeds(); /* Halt in case of failure */
    }
    else if (bsec.sensor.status > BME68X_OK)
    {
        Serial.println("BME68X warning code : " + String(bsec.sensor.status));
    }
}
