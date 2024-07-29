#include <TFT_eSPI.h>
#include <Adafruit_Fingerprint.h>
#include <PubSubClient.h>
#include <WiFi.h>

#include "secrets.h"
#include "logo/logo.h"
#include "logo/success.h"
#include "logo/wifi.h"
#include "logo/error.h"


/****************************
 *     Global Variables     *
 ****************************/
TFT_eSPI tft = TFT_eSPI();

HardwareSerial mySerial(1); // Crea un oggetto Serial per UART1
Adafruit_Fingerprint finger = Adafruit_Fingerprint(&mySerial);
int steps_enroll = 1;
int attempts = 0;
bool closed = false;

WiFiClient espClient;
PubSubClient mqtt_client(espClient);


/********************************
 * functions fingerprint_module *
 ********************************/
void fingerprint_setup();
uint8_t fingerprint_match();


/*******************************
 * functions connection_module *
 *******************************/
void connectToWiFi();
void connectToMQTT();
void mqttCallback(char *topic, byte *payload, unsigned int length);
