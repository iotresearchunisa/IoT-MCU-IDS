/***********************************************************************************************************
* - METRICHE STATISTICHE GENERALI                                                                          *
*     - [Accuracy]: Percentuale di predizioni corrette.                                                    *
*     - [Tempo Medio per Predizione]: Tempo medio impiegato per eseguire una predizione.                   *
*     - [Deviazione Standard del Tempo]: Variazione dei tempi di predizione rispetto alla media.           *
*     - [Tempo Minimo e Massimo per Predizione]: I limiti inferiori e superiori dei tempi di predizione.   *
************************************************************************************************************


************************************************************************************************************
* - METRICHE TEMPORALI SPECIFICHE DELLE OPERAZIONI                                                         *
      - [Caricamento degli Input]:                                                                         *
          - Tempo Medio: Tempo medio impiegato per caricare gli input.                                     *
          - Deviazione Standard: Variazione dei tempi di caricamento.                                      *
          - Tempo Minimo e Massimo: I limiti inferiori e superiori dei tempi di caricamento.               *
                                                                                                           *
      - [Inferenza del Modello]:                                                                           *
          - Tempo Medio: Tempo medio impiegato per eseguire l'inferenza.                                   *
          - Deviazione Standard: Variazione dei tempi di inferenza.                                        *
          - Tempo Minimo e Massimo: I limiti inferiori e superiori dei tempi di inferenza.                 *
                                                                                                           *
      - [Post-elaborazione]:                                                                               *
          - Tempo Medio: Tempo medio impiegato per la post-elaborazione.                                   *
          - Deviazione Standard: Variazione dei tempi di post-elaborazione.                                *
          - Tempo Minimo e Massimo: I limiti inferiori e superiori dei tempi di post-elaborazione.         *
************************************************************************************************************/


void loop() {
  int correct_predictions = 0;
  int total_predictions = 0;

  total_stats = TimingStats();
  load_stats = TimingStats();
  inference_stats = TimingStats();
  postprocess_stats = TimingStats();


  for (int i = 0; i < NUM_PAIRS; i++) {
    unsigned long pair_start_time = micros(); //Inizio della predizione

    // Misura il tempo di caricamento degli input
    unsigned long load_start = micros();
    load_inputs(i);
    unsigned long load_end = micros();
    unsigned long load_duration = load_end - load_start;
    updateTimingStats(load_stats, load_duration);


    // Misura il tempo di inferenza
    unsigned long inference_start = micros();
    float similarity = run_inference();
    unsigned long inference_end = micros(); 
    unsigned long inference_duration = inference_end - inference_start;
    updateTimingStats(inference_stats, inference_duration);


    // Misura il tempo di post-elaborazione (calcolo della predizione e verifica)
    unsigned long postprocess_start = micros();
    int real_label = SIAMESE_MODEL_LABELS[i];
    int predicted_label = (similarity < 0.5f) ? 1 : 0;

    if (predicted_label == real_label) {
        correct_predictions++;
    }

    total_predictions++;
    unsigned long postprocess_end = micros();
    unsigned long postprocess_duration = postprocess_end - postprocess_start;
    updateTimingStats(postprocess_stats, postprocess_duration);


    // Calcolo del tempo totale per questa coppia
    unsigned long pair_end_time = micros();
    unsigned long pair_duration = pair_end_time - pair_start_time;
    updateTimingStats(total_stats, pair_duration);


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

    //delay(500);
  }

  // Calcola le statistiche generali
  calculateStatistics(total_stats, NUM_PAIRS);

  // Calcola le statistiche specifiche delle operazioni
  calculateStatistics(load_stats, NUM_PAIRS);
  calculateStatistics(inference_stats, NUM_PAIRS);
  calculateStatistics(postprocess_stats, NUM_PAIRS);

  // Calcola e stampa l'accuratezza
  float accuracy = (total_predictions > 0) ? ((float)correct_predictions / total_predictions) * 100.0f : 0.0f;
  Serial.print("Accuracy: ");
  Serial.print(accuracy, 2);
  Serial.println("%");

  // Stampa delle statistiche temporali in secondi con 4 cifre decimali
  printTimingStats("Tempo medio per predizione", total_stats);
  printTimingStats("Tempo medio di caricamento", load_stats);
  printTimingStats("Tempo medio di inferenza", inference_stats);
  printTimingStats("Tempo medio di post-elaborazione", postprocess_stats);

  // Stampa dei tempi minimo e massimo separatamente per maggiore chiarezza
  Serial.print("Tempo minimo per predizione: ");
  Serial.print(total_stats.min_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo massimo per predizione: ");
  Serial.print(total_stats.max_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo minimo di caricamento: ");
  Serial.print(load_stats.min_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo massimo di caricamento: ");
  Serial.print(load_stats.max_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo minimo di inferenza: ");
  Serial.print(inference_stats.min_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo massimo di inferenza: ");
  Serial.print(inference_stats.max_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo minimo di post-elaborazione: ");
  Serial.print(postprocess_stats.min_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.print("Tempo massimo di post-elaborazione: ");
  Serial.print(postprocess_stats.max_time / 1000000.0, 8);
  Serial.println(" secondi");

  Serial.println("Test completato.");
  while(true){}
}
