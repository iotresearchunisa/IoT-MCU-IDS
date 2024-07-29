#include <TFT_eSPI.h>
#include <Adafruit_Fingerprint.h>
#include <PubSubClient.h>
#include <time.h>
#include "src/dependencies/WiFiClientSecure/WiFiClientSecure.h" //using older WiFiClientSecure

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

// NTP Server settings
const char *ntp_server = "192.168.14.240";
const long gmt_offset_sec = 1L * 60L * 60L;            // GMT offset in seconds (adjust for your time zone)
const int daylight_offset_sec = 3600;                  // Daylight saving time offset in seconds

// WiFi and MQTT client initialization
WiFiClientSecure espClient;
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
void syncTime();
void mqttCallback(char *topic, byte *payload, unsigned int length);
