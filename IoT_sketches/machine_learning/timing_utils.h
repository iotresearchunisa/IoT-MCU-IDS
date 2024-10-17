#ifndef TIMING_UTILS_H
#define TIMING_UTILS_H

#include <Arduino.h>
#include <limits.h>

// Struttura per tenere traccia delle statistiche temporali
struct TimingStats {
    unsigned long total_time;
    unsigned long squared_time;
    unsigned long min_time;
    unsigned long max_time;
    float mean_time;
    float std_dev_time;
    
    // Costruttore per inizializzare i valori
    TimingStats() : total_time(0), squared_time(0), min_time(ULONG_MAX), max_time(0),
                   mean_time(0.0f), std_dev_time(0.0f) {}
};

// Funzione per aggiornare le statistiche temporali
void updateTimingStats(TimingStats &stats, unsigned long duration);

// Funzione per calcolare la media e la deviazione standard
void calculateStatistics(TimingStats &stats, int num_samples);

// Funzione per stampare le statistiche temporali in secondi con 4 cifre decimali
void printTimingStats(const char* label, TimingStats &stats);

#endif // TIMING_UTILS_H
