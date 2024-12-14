#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "test_pairs.h" 
#include "timing_utils.h"

#define PIN 4 


namespace {
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;

  TfLiteTensor *input_a = nullptr;
  TfLiteTensor *input_b = nullptr;
  TfLiteTensor *output = nullptr;

  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// Variabili per le statistiche generali
TimingStats total_stats;

// Variabili per le statistiche specifiche delle operazioni
TimingStats load_stats;
TimingStats inference_stats;
TimingStats postprocess_stats;

void setup_model();
void load_inputs(int);
float run_inference();