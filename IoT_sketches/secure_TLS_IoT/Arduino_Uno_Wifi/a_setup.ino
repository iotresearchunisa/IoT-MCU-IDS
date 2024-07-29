void setup() {
  Serial.begin(9600);

  /*********
  * Buzzer *
  **********/
  pinMode(PIN_BUZZER, OUTPUT);


  /*********
  * Sensor *
  **********/
  pinMode(PIN_SENSOR, INPUT);
  digitalWrite(PIN_SENSOR, LOW);
  //calibrate();


  /****************
  *   Connection  *
  *****************/
  connectToWiFi();
  timeClient.begin();
  syncTime();

  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqttCallback);

  connectToMQTT();
}
