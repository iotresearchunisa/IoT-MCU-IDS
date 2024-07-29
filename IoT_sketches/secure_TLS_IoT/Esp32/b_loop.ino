
void loop(){

  if (WiFi.status() == WL_CONNECTED) {

    if (!mqtt_client.connected()) {
      connectToMQTT();
    }

    mqtt_client.loop();

    if(!closed) {
      tft.setCursor(5, 47);
      tft.print("Autentication:");
      tft.setCursor(5, 74);
      tft.print(attempts);
      tft.print("/3");

      int result = fingerprint_match();

      if (result > 0 && result != 100) {
        tft.fillScreen(TFT_BLACK);
        tft.setCursor(43, 60);
        tft.print("Autenticated !!");
        tft.pushImage(90, 75, 52, 52, success);
        
        mqtt_client.publish(MQTT_PUB_ESP8266, "autenticato_esp32");

        attempts = 0;
        delay(5000);
        tft.fillScreen(TFT_BLACK);
      } 
      else if (result == 100) {
        tft.fillScreen(TFT_BLACK);
        attempts += 1; 

        tft.setCursor(5, 47);
        tft.print("Invio dati:");
        tft.setCursor(5, 74);
        tft.print("attendi...");

        mqtt_client.publish(MQTT_PUB_ESP8266, "tentativo_di_accesso_esp32");
        delay(5000);   
        tft.fillScreen(TFT_BLACK);
      } 

      delay(100); 

      if(attempts >= 3){
        tft.fillScreen(TFT_BLACK);
        tft.setCursor(43, 60);
        tft.print("VIOLATION !");
        tft.pushImage(90, 75, 52, 52, error);

        mqtt_client.publish(MQTT_PUB_ESP8266, "autenticazione_fallita_esp32");
        
        closed = true;
        attempts = 0;
        delay(3000);
        tft.fillScreen(TFT_BLACK);
      }
    }
    else {
      tft.setCursor(43, 60);
      tft.print("Close Door !");
      tft.pushImage(90, 75, 52, 52, error);

      attempts = 0;
    }  
  }
  else {
    connectToWiFi();
  }
}
