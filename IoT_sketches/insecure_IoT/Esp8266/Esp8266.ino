#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include "secrets.h"


/****************************
 *     Global Variables     *
 ****************************/
WiFiClient espClient;
PubSubClient mqtt_client(espClient);


/**********************************
 * functions communication_module *
 **********************************/
void connectToWiFi();
void connectToMQTT();
void mqttCallback(char *topic, byte *payload, unsigned int length);
void see_esp8266_topic(char *message);