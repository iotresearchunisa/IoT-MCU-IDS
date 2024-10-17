// Funzione per inizializzare il modello
void setup_model(){
  // Load Model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
      "Model provided is schema version %d not equal to supported "
      "version %d.",
      model->version(), TFLITE_SCHEMA_VERSION
    );
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddRelu();
  resolver.AddLogistic();
  resolver.AddMaxPool2D();
  resolver.AddReshape();

  resolver.AddSquaredDifference();
  resolver.AddSum();
  resolver.AddMaximum();
  resolver.AddSqrt();

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input_a = interpreter->input(0);
  input_b = interpreter->input(1);
  output = interpreter->output(0);
}


// Funzione per caricare gli input
void load_inputs(int pair_index) {
    // Assicurati che pair_index sia valido
    if (pair_index >= NUM_PAIRS) {
        Serial.println("Indice coppia fuori range!");
        return;
    }

    // Carica i dati di SIAMESE_MODEL_pairs_a e SIAMESE_MODEL_pairs_b negli input
    for (int i = 0; i < FEATURE_SIZE; i++) {
        input_a->data.f[i] = pairs_a[pair_index][i][0][0];
        input_b->data.f[i] = pairs_b[pair_index][i][0][0];
    }
}


// Funzione per eseguire l'inferenza
float run_inference() {
    // Esegui l'inferenza
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inferenza fallita!");
        return -1.0f;
    }

    // Ottieni l'output
    float similarity = output->data.f[0];
    return similarity;
}