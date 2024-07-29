void connectToWiFi() {
  WiFi.mode(WIFI_STA); 
  WiFi.begin(ssid, password);

  // attempt to connect to Wifi network:
  unsigned long start_time_connection = millis();
  do {
    tft.setCursor(5, 47);
    tft.print("Connecting to");
    tft.setCursor(5, 74);
    tft.print(ssid);
    
    delay(400); tft.print(".");
    delay(400); tft.print(".");
    delay(400); tft.print(".");
    delay(400);
    tft.fillRect(0,25,235,110,TFT_BLACK);
    
    if((millis() - start_time_connection) > 15000){
      tft.setCursor(5, 47);
      tft.print("HotSpot not");
      tft.setCursor(5, 74);
      tft.print("found.");
      
      delay(3000);
      tft.fillRect(0,25,235,110,TFT_BLACK);
      return;
    }
  } while (WiFi.status() != WL_CONNECTED);
  
  tft.setCursor(5, 47);
  tft.fillRect(0,25,235,110,TFT_BLACK);
  tft.print("Connected to ");
  tft.setCursor(5, 74);
  tft.print(ssid); 
  tft.pushImage(90, 84, 60, 48, wifi);
  delay(3000);
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
  while (!mqtt_client.connected()){
    if (mqtt_client.connect(HOSTNAME, MQTT_USER, MQTT_PASS)) {
    tft.setCursor(5, 47);
    tft.print("Connected to");
    tft.setCursor(5, 74);
    tft.print("MQTT broker.");
    delay(3000);
    tft.fillRect(0,25,235,110,TFT_BLACK);

    mqtt_client.subscribe(MQTT_SUB_ESP32);
    } 
    else {
      tft.setCursor(5, 47);
      tft.print("Failed to connect");
      tft.setCursor(5, 74);
      tft.print("MQTT broker.");
      delay(4000);
      tft.fillRect(0,25,235,110,TFT_BLACK);
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

  if(strcmp(topic, "esp32/topic") == 0){
    see_esp32_topic(message);
  }

  Serial.println(message);
}


void see_esp32_topic(char *message){
  // esp8266 --> esp32
  if(strcmp(message, "blocca") == 0){
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(43, 60);
    tft.print("Close Door !");
    tft.pushImage(90, 75, 52, 52, error);

    attempts = 0;
    closed = true;
  } 
  else if (strcmp(message, "apri porta") == 0) {
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(43, 60);
    tft.print("Open Door !!");
    tft.pushImage(90, 75, 52, 52, success);

    attempts = 0;
    closed = false;
    delay(4000);
    tft.fillScreen(TFT_BLACK);
  }
}
