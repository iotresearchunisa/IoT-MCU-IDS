const char ssid[] = "Rasberry-AP";
//const char password[] = "Password123";

#define HOSTNAME "esp32"

IPAddress MQTT_BROKER(192, 168, 14, 240);
const int MQTT_PORT = 1883;

const char MQTT_SUB_ESP32[] = "esp32/topic";
const char MQTT_PUB_ESP8266[] = "esp8266/topic";
