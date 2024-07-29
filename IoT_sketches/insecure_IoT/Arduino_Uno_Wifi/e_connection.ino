void connectToWiFi() {
  
  // check for the WiFi module:
  if (WiFi.status() == WL_NO_MODULE) {
    Serial.println("Communication with WiFi module failed!");
    // don't continue
    while (true);
  }

  String fv = WiFi.firmwareVersion();
  if (fv < WIFI_FIRMWARE_LATEST_VERSION) {
    Serial.println("Please upgrade the firmware");
  }

  // attempt to connect to WiFi network:
  while (status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);

    //status = WiFi.begin(ssid, password);
    status = WiFi.begin(ssid);
    delay(5000);
  }

  Serial.println();
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.println();
}


void connectToMQTT() {
  while (!mqtt_client.connected()) {
    if (mqtt_client.connect(HOSTNAME)) {
      Serial.println("Connected to MQTT broker");

      mqtt_client.subscribe(MQTT_SUB_ARDUINO);
    } 
    else {
      Serial.print("Failed to connect to MQTT broker, rc=");
      Serial.println(mqtt_client.state());
      delay(5000);
    }
  }
}


void mqttCallback(char *topic, byte *payload, unsigned int length) {
  Serial.print("Message received on topic: ");
  Serial.print(topic);
  Serial.print("--> ");

  char message[length];
    
  for (int i = 0; i < length; i++) {
    message[i] = (char) payload[i];
  }
  message[length] = '\0';

   if(strcmp(topic, "arduino/topic") == 0)
      see_arduino_topic(message);
  
  Serial.println(message);
}


void see_arduino_topic(char *message){
  if(strcmp(message, "allarme") == 0){
    alarm();
  }
}
