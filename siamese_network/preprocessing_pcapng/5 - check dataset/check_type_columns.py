import pandas as pd


def process_csv_in_chunks(file_path, chunk_size=10000):
    unique_values = set()  # Insieme per valori unici della colonna 0
    dtypes = None  # Dizionario per i tipi di dati delle colonne

    # Leggi il file in chunk
    for chunk in pd.read_csv(file_path, delimiter=';', chunksize=chunk_size, low_memory=False):
        # Ottieni valori unici nella colonna 0 (prima colonna)
        unique_values.update(chunk.iloc[:, 0].unique())

        # Determina i tipi di dati di tutte le colonne, solo una volta
        if dtypes is None:
            dtypes = chunk.dtypes

    # Output dei risultati
    print(f"Valori unici della colonna 0: {unique_values}")
    print("Tipi di dati di tutte le colonne:")
    print(dtypes)


# Specifica il percorso del file CSV e la dimensione del chunk
process_csv_in_chunks('/mnt/FE9090E39090A3A5/Tesi/TON_IoT/ton_iot_dataset.csv', chunk_size=300000)
