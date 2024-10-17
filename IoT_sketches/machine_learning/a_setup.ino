void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("Inizializzazione del modello Siamese...");
 
  setup_model();
  Serial.println("Modello inizializzato correttamente!");
}
