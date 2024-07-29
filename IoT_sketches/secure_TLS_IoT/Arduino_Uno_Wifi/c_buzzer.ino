/* ALARM 1 */
void alarm() {
  unsigned long  start_time = millis();
  
  while(millis()  - start_time < 6000) {
    tone(PIN_BUZZER, millis()%1000+200);
  }

  noTone(PIN_BUZZER);
}
/* ALARM 1 */


/* ALARM 2 */
void frequenza_1() {
  for(int i = 0; i < 80; i++) {
    digitalWrite(PIN_BUZZER,HIGH);
    delay(1);
    
    digitalWrite(PIN_BUZZER,LOW);
    delay(1);
  }
}

void frequenza_2() {
  for(int i=0;i<100;i++) {
    digitalWrite(PIN_BUZZER, HIGH);
    delay(2);
        
    digitalWrite(PIN_BUZZER, LOW);
    delay(2);
  }
}

void alarm_2() {
  frequenza_1();
  frequenza_2();
}
/* ALARM 2 */
