#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "test_pairs.h" 


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input_a = nullptr;
TfLiteTensor *input_b = nullptr;
TfLiteTensor *output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 101 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  Serial.begin(115200);
  delay(2000);
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
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
  input_a = interpreter->input(0);  // Primo input
  input_b = interpreter->input(1);  // Secondo input
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  static int current_pair = 0;
  static int correct_predictions = 0;

  if (current_pair >= NUM_PAIRS) {
    // Tutti i test sono stati eseguiti, stampa l'accuratezza totale
    float accuracy = (float)correct_predictions / NUM_PAIRS;
    Serial.printf("Accuracy totale: %.2f%%\n", accuracy * 100);
    while (true) {
      // Blocca il loop dopo aver stampato l'accuratezza
      delay(1000);
    }
  }

  // Reset degli input
  memset(input_a->data.f, 0, input_a->bytes);
  memset(input_b->data.f, 0, input_b->bytes);

  // Carica i dati di SIAMESE_MODEL_pairs_a e SIAMESE_MODEL_pairs_b negli input
  for (int i = 0; i < FEATURE_SIZE; i++) {
    input_a->data.f[i] = pairs_a[current_pair][i][0][0];
    input_b->data.f[i] = pairs_b[current_pair][i][0][0];
  }

  // Esegui l'inferenza
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on pair %d", current_pair);
    return;
  }

  // Ottieni la predizione
  float prediction = output->data.f[0];
  int predicted_label = (prediction < 0.5) ? 1 : 0;  // Adatta la soglia se necessario
  int actual_label = SIAMESE_MODEL_LABELS[current_pair];

  // Confronta e conta le corrette
  if (predicted_label == actual_label) {
    correct_predictions++;
  }

  // Stampa la predizione e l'etichetta reale
  Serial.printf("Coppia %d: Predizione = %d, Reale = %d\n", current_pair, predicted_label, actual_label);

  // Passa alla coppia successiva
  current_pair++;
  
  // Aggiungi un breve delay per evitare di sovraccaricare il seriale
  delay(10);

  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  /*
  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) {
    inference_count = 0;
  }*/
}
