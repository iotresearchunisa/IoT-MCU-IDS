
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
  syncTime();  // X.509 validation requires synchronization time

  espClient.setCACert(ca_cert);          //Root CA certificate
  espClient.setCertificate(client_cert); //for client verification if the require_certificate is set to true in the mosquitto broker config
  espClient.setPrivateKey(client_key);  //for client verification if the require_certificate is set to true in the mosquitto broker config

  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqttCallback);
  tft.fillScreen(TFT_BLACK);
  connectToMQTT();

  tft.fillScreen(TFT_BLACK);
}
