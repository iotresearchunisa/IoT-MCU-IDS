uint8_t getFingerprintID() {
  uint8_t p = finger.getImage();

  switch (p) {
    case FINGERPRINT_OK:
      break;
      
    case FINGERPRINT_NOFINGER:
      return 0;
      
    case FINGERPRINT_PACKETRECIEVEERR:
      return 0;
      
    case FINGERPRINT_IMAGEFAIL:
      return 0;
      
    default:
      return 0;
  }

  // OK success!
  p = finger.image2Tz();
  switch (p) {
    case FINGERPRINT_OK:
      break;
      
    case FINGERPRINT_IMAGEMESS:
      return 0;
      
    case FINGERPRINT_PACKETRECIEVEERR:
      return 0;
      
    case FINGERPRINT_FEATUREFAIL:
      return 0;
      
    case FINGERPRINT_INVALIDIMAGE:
      return 0;
      
    default:
      return 0;
  }

  // OK converted!
  p = finger.fingerSearch();
  if (p == FINGERPRINT_OK) {} 
  
  else if (p == FINGERPRINT_PACKETRECIEVEERR) 
    return 0;
    
  else if (p == FINGERPRINT_NOTFOUND)
    return 100;

  else
    return 0;

  return finger.fingerID;
}


void fingerprint_setup(){
  while (!Serial);
  delay(100);
  
  finger.begin(57600);
  delay(5);
  
  if (finger.verifyPassword()) {
    Serial.println("Found fingerprint sensor!");
  } else {
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(5, 47);
    tft.print("Fingerprint don't");
    tft.setCursor(5, 80);
    tft.print("detect.");
    tft.setCursor(5, 122);
    tft.print("Turn off the power."); 
    while (1) { delay(1); }
  }
}


uint8_t fingerprint_match(){
  finger.getTemplateCount();

  if (finger.templateCount == 0)
    return 0;

  return getFingerprintID();
}