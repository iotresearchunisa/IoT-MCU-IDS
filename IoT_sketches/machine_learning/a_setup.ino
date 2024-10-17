void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("Inizializzazione del modello Siamese...");
 
  setup_model();
  Serial.println("Modello inizializzato correttamente!");
}
