#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <time.h>
#include "secrets.h"


/****************************
 *     Global Variables     *
 ****************************/
// NTP Server settings
const char *ntp_server = "192.168.14.240";
const long gmt_offset_sec = 1L * 60L * 60L;            // GMT offset in seconds (adjust for your time zone)
const int daylight_offset_sec = 3600;                  // Daylight saving time offset in seconds

// WiFi and MQTT client initialization
BearSSL::WiFiClientSecure espClient;
PubSubClient mqtt_client(espClient);


/**********************************
 * functions communication_module *
 **********************************/
void connectToWiFi();
void connectToMQTT();
void syncTime();
void mqttCallback(char *topic, byte *payload, unsigned int length);
void see_esp8266_topic(char *message);