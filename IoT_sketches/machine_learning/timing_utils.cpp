#include "timing_utils.h"
#include <math.h>

// Aggiorna le statistiche con una nuova durata
void updateTimingStats(TimingStats &stats, unsigned long duration) {
  stats.total_time += duration;
  stats.squared_time += ((unsigned long)duration * duration);
  
  if (duration < stats.min_time) {
      stats.min_time = duration;
  }
  if (duration > stats.max_time) {
      stats.max_time = duration;
  }
}

// Calcola la media e la deviazione standard
void calculateStatistics(TimingStats &stats, int num_samples) {
  if (num_samples > 0) {
      stats.mean_time = (float)stats.total_time / num_samples;
      float variance = ((float)stats.squared_time / num_samples) - (stats.mean_time * stats.mean_time);
      
      if (variance < 0.0f) {
          variance = 0.0f;
      }

      stats.std_dev_time = sqrt(variance);
  }
}

// Stampa le statistiche temporali
void printTimingStats(const char* label, TimingStats &stats) {
  Serial.print(label);
  Serial.print(": ");
  Serial.print(stats.mean_time / 1000000.0, 8); // Converti da microsecondi a secondi
  Serial.print(" sec | Std Dev: ");
  Serial.print(stats.std_dev_time / 1000000.0, 8);
  Serial.print(" sec | Min: ");
  Serial.print(stats.min_time / 1000000.0, 8);
  Serial.print(" sec | Max: ");
  Serial.print(stats.max_time / 1000000.0, 8);
  Serial.println(" sec");
}
