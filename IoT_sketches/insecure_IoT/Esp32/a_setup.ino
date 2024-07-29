
void setup(){
  Serial.begin(115200);
  mySerial.begin(115200, SERIAL_8N1, 27, 26); // RX=27, TX=26
    
  tft.begin();
  tft.setRotation(1);
  tft.fillScreen(TFT_BLACK);
  tft.setSwapBytes(true);
  tft.setTextColor(tft.color565(3, 211, 216), TFT_BLACK);
  tft.setFreeFont(&FreeSans12pt7b);
  tft.pushImage(56, 18, 140, 99, logo);
  
  delay(3000);
  tft.fillScreen(TFT_BLACK);

  fingerprint_setup();
  connectToWiFi();

  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqttCallback);
  tft.fillScreen(TFT_BLACK);
  connectToMQTT();

  tft.fillScreen(TFT_BLACK);
}
