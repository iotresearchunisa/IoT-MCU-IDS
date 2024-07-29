void loop() {
  if (WiFi.status() == WL_CONNECTED) {

    if (!mqtt_client.connected()) {
      connectToMQTT();
    }
    mqtt_client.loop();
  }
  else{
    connectToWiFi();
  }
}