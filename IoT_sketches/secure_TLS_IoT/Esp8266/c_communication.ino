void connectToWiFi() {
  WiFi.mode(WIFI_STA); 
  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");
  
  while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  Serial.println();
}


void syncTime() {
    configTime(gmt_offset_sec, daylight_offset_sec, ntp_server);
    Serial.print("Waiting for NTP time sync: ");
    
    while (time(nullptr) < 8 * 3600 * 2) {
        delay(1000);
        Serial.print(".");
    }

    Serial.println("Time synchronized");
    struct tm timeinfo;

    if (getLocalTime(&timeinfo)) {
        Serial.print("Current time: ");
        Serial.println(asctime(&timeinfo));
    } else {
        Serial.println("Failed to obtain local time");
    }
}


void connectToMQTT() {
  while (!mqtt_client.connected()) {
    if (mqtt_client.connect(HOSTNAME, MQTT_USER, MQTT_PASS)) {
      Serial.println("Connected to MQTT broker");

      mqtt_client.subscribe(MQTT_SUB_ESP82);
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

   if(strcmp(topic, "esp8266/topic") == 0)
      see_esp8266_topic(message);
  
  Serial.println(message);
}


void see_esp8266_topic(char *message){

  if(strcmp(message, "autenticato_esp32") == 0){
    mqtt_client.publish(MQTT_PUB_RASBERRY, "autenticato");
  } 
  else if (strcmp(message, "tentativo_di_accesso_esp32") == 0) {
    mqtt_client.publish(MQTT_PUB_RASBERRY, "tentativo_di_accesso");
  } 
  else if (strcmp(message, "autenticazione_fallita_esp32") == 0) {
    mqtt_client.publish(MQTT_PUB_RASBERRY, "autenticazione_fallita");
    mqtt_client.publish(MQTT_PUB_ARDUINO, "allarme");
  }
  else if(strcmp(message, "persona_rilevata_arduino") == 0){
    mqtt_client.publish(MQTT_PUB_RASBERRY, "persona_rilevata");
  } 
  else if(strcmp(message, "persona_rilevata_end_arduino") == 0){
    mqtt_client.publish(MQTT_PUB_RASBERRY, "persona_rilevata_end");
  }  
  else if (strcmp(message, "porta_aperta") == 0) {
    mqtt_client.publish(MQTT_PUB_ESP32, "apri porta");
  } 
  else if (strcmp(message, "blocca") == 0) {
    mqtt_client.publish(MQTT_PUB_ESP32, "blocca");
  } 
  else if (strcmp(message, "allarme") == 0) {
    mqtt_client.publish(MQTT_PUB_ARDUINO, "allarme");
  } 
}