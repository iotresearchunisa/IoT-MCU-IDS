#include <SPI.h>
#include <WiFiNINA.h>
#include <PubSubClient.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <time.h>
#include "secrets.h"

#define PIN_BUZZER 8
#define PIN_SENSOR 9


/****************************
 *     Time Variables     *
 ****************************/
const char *ntp_server = "192.168.14.240";
const long utcOffsetInSecondsStandard = 3600; // UTC+1 per CET
const long utcOffsetInSecondsDaylight = 7200; // UTC+2 per CEST

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, ntp_server, utcOffsetInSecondsStandard);


/***********************
*   Sensor variables   *
************************/
int calibrationTime = 30; // the time we give the sensor to calibrate (10-60 secs according to the datasheet)  
long unsigned int lowIn;  // the time when the sensor outputs a low impulse
long unsigned int pause_d = 10000;  // the amount of milliseconds the sensor has to be low before we assume all motion has stopped

boolean lockLow = true;
boolean takeLowTime = true;   


/*********************
*   MQTT variables   *
**********************/
WiFiSSLClient arduinoClient;
PubSubClient mqtt_client(arduinoClient);
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
void syncTime();
