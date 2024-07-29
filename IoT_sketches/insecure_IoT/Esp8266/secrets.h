const char ssid[] = "Rasberry-AP";
//const char password[] = "Password123";

#define HOSTNAME "esp8266"

IPAddress MQTT_BROKER(192, 168, 14, 240);
const int MQTT_PORT = 1883;

const char MQTT_SUB_ESP82[] = "esp8266/topic";

const char MQTT_PUB_RASBERRY[] = "rasberry/topic";
const char MQTT_PUB_ESP32[] = "esp32/topic";
const char MQTT_PUB_ARDUINO[] = "arduino/topic";
