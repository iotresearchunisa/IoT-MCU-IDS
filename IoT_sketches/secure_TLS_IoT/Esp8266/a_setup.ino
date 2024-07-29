void setup() {
  Serial.begin(115200);

  connectToWiFi();
  syncTime();  // X.509 validation requires synchronization time

  espClient.setTrustAnchors(new BearSSL::X509List(ca_cert));

  /* 
    Se nel file 'sudo nano /etc/mosquitto/conf.d/mosquitto.conf' si imposta
    'require_certificate false', commenta questa riga 
  */
  espClient.setClientRSACert(new BearSSL::X509List(client_cert), new BearSSL::PrivateKey(client_key));

  mqtt_client.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt_client.setCallback(mqttCallback);
  connectToMQTT();
}
