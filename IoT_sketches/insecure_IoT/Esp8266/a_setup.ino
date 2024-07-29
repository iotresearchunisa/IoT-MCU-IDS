void setup() {
  Serial.begin(115200);
  Serial.println();
  
  connectToWiFi();
  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqttCallback);
  connectToMQTT();
}
