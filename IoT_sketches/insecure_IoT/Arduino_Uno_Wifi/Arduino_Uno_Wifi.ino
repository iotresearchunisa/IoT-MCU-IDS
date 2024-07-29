#include <SPI.h>
#include <WiFiNINA.h>
#include <PubSubClient.h>
#include "secrets.h"

#define PIN_BUZZER 8
#define PIN_SENSOR 9


/***********************
*   Sensor variables   *
************************/
int calibrationTime = 30; // the time we give the sensor to calibrate (10-60 secs according to the datasheet)  
long unsigned int lowIn;  // the time when the sensor outputs a low impulse
long unsigned int pause = 10000;  // the amount of milliseconds the sensor has to be low before we assume all motion has stopped

boolean lockLow = true;
boolean takeLowTime = true;   


/*********************
*   MQTT variables   *
**********************/
WiFiClient espClient;
PubSubClient mqtt_client(espClient);
int status = WL_IDLE_STATUS;


/*************
*   Buzzer   *
**************/
void alarm();
void alarm_2();


/************
*   Sensor  *
*************/
void calibrate();
void detects_movement();


/****************
*   Connection  *
*****************/
void connectToWiFi();
void connectToMQTT();
void mqttCallback(char *topic, byte *payload, unsigned int length);
void see_arduino_topic(char *message);
