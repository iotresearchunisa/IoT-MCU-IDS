void calibrate() {
  Serial.print("calibrating sensor ");
  
  for(int i = 0; i < calibrationTime; i++){
    Serial.print(".");
    delay(1000);
  }

  Serial.println(" done");
  Serial.println("SENSOR ACTIVE");
  delay(50);
}


void detects_movement(){
  if(digitalRead(PIN_SENSOR) == HIGH){
    if(lockLow){  
      // makes sure we wait for a transition to LOW before any further output is made:
      lockLow = false;            
      Serial.println("---");
      Serial.print("motion detected at ");
      Serial.print(millis()/1000);
      Serial.println(" sec"); 

      mqtt_client.publish(MQTT_PUB_ESP82, "persona_rilevata_arduino");
      delay(50);
    }         
    takeLowTime = true;
  }

  if(digitalRead(PIN_SENSOR) == LOW){       
    if(takeLowTime){
      lowIn = millis();          //save the time of the transition from high to LOW
      takeLowTime = false;       //make sure this is only done at the start of a LOW phase
    }

  //if the sensor is low for more than the given pause, 
  //we assume that no more motion is going to happen
  if(!lockLow && millis() - lowIn > pause_d){  
      //makes sure this block of code is only executed again after 
      //a new motion sequence has been detected
      lockLow = true;                        
      Serial.print("motion ended at ");
      Serial.print((millis() - pause_d)/1000);
      Serial.println(" sec");
      
      mqtt_client.publish(MQTT_PUB_ESP82, "persona_rilevata_end_arduino");
      delay(50);
    }
  }
}
