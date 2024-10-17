void loop() {
  int correct_predictions = 0;
  int total_predictions = 0;

  for (int i = 0; i < NUM_PAIRS; i++) {
      // Carica gli input della coppia corrente
      load_inputs(i);

      // Esegui l'inferenza
      float similarity = run_inference();

      // Ottieni la label reale
      int real_label = SIAMESE_MODEL_LABELS[i];

      // Decidi una soglia per classificare come simile o diverso
      int predicted_label = (similarity < 0.5f) ? 1 : 0;

      // Verifica se la predizione è corretta
      if (predicted_label == real_label) {
          correct_predictions++;
      }

      total_predictions++;

      // Stampa i risultati
      Serial.print("Coppia ");
      Serial.print(i);
      Serial.print(": Similarità = ");
      Serial.print(similarity, 6);
      Serial.print(" | Predizione = ");
      Serial.print(predicted_label);
      Serial.print(" | Reale = ");
      Serial.println(real_label);

      // Reset degli input per sicurezza (opzionale)
      // memset(input1->data.f, 0, input1->bytes);
      // memset(input2->data.f, 0, input2->bytes);

      delay(500);
  }

  // Calcola e stampa l'accuratezza
  float accuracy = (total_predictions > 0) ? ((float)correct_predictions / total_predictions) * 100.0f : 0.0f;
  Serial.print("Accuracy: ");
  Serial.print(accuracy, 2);
  Serial.println("%");

  Serial.println("Test completato.");
  while(true){}
}
