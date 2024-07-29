// Funzione per controllare se è ora legale
bool isDaylightSavingTime(struct tm * timeinfo) {
  // Regole per l'ora legale in Europa
  // Inizia l'ultima domenica di marzo
  // Termina l'ultima domenica di ottobre
  if (timeinfo->tm_mon > 2 && timeinfo->tm_mon < 9) {
    return true;  // da aprile a settembre
  }
  if (timeinfo->tm_mon == 2 && (timeinfo->tm_wday + 31 - timeinfo->tm_mday) < 7) {
    return true;  // ultima settimana di marzo
  }
  if (timeinfo->tm_mon == 9 && (timeinfo->tm_wday + 31 - timeinfo->tm_mday) < 7) {
    return false;  // ultima settimana di ottobre
  }
  return false;
}


void syncTime() {
  // Aggiorna e sincronizza l'orario
  Serial.print("Waiting for NTP time sync: ");
  
  while(!timeClient.update()) {
    timeClient.forceUpdate();
    delay(1000);
    Serial.print(".");
  }

  Serial.println("Time synchronized");
  
  // Ottieni l'ora corrente
  time_t rawtime = timeClient.getEpochTime();
  struct tm * timeinfo = gmtime(&rawtime);
  
  // Imposta l'offset per l'ora legale se necessario
  if (isDaylightSavingTime(timeinfo)) {
    timeClient.setTimeOffset(utcOffsetInSecondsDaylight);
  } else {
    timeClient.setTimeOffset(utcOffsetInSecondsStandard);
  }

  // Aggiorna di nuovo per applicare l'offset
  timeClient.update();

  Serial.print("Current time: ");
  Serial.println(timeClient.getFormattedTime());
}