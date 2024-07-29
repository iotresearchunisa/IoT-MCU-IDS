const char ssid[] = "Rasberry-AP";
//const char password[] = "Password123";

#define HOSTNAME "arduino_uno_wifi_rev2"

IPAddress MQTT_BROKER(192, 168, 14, 240);
const int MQTT_PORT = 1883;

const char MQTT_SUB_ARDUINO[] = "arduino/topic";
const char MQTT_PUB_ESP82[] = "esp8266/topic";

