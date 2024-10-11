import pandas as pd
import json
import sys
import os


def create_protocol_counts(csv_file_path, chunk_size=100000):
    """
    Legge un file CSV in chunks e conta le occorrenze dei valori unici nella colonna 'Protocol_Type'.

    Parameters:
    - csv_file_path: str, percorso al file CSV.
    - chunk_size: int, numero di righe per ogni chunk (default: 100000).

    Returns:
    - protocol_counts: dict, conteggio delle occorrenze per ogni protocollo.
    """
    protocol_counts = {}

    try:
        # Legge il CSV in chunks
        for chunk in pd.read_csv(csv_file_path, delimiter=';', chunksize=chunk_size):
            if 'Protocol_Type' not in chunk.columns:
                raise ValueError("La colonna 'Protocol_Type' non esiste nel file CSV.")

            # Conta le occorrenze di ogni Protocol_Type nel chunk
            counts = chunk['Protocol_Type'].value_counts()
            for protocol, count in counts.items():
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + count
    except FileNotFoundError:
        print(f"Errore: Il file '{csv_file_path}' non è stato trovato.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Errore: Il file CSV è vuoto.")
        sys.exit(1)
    except Exception as e:
        print(f"Si è verificato un errore: {e}")
        sys.exit(1)

    return protocol_counts


def create_mapping(protocol_list):
    """
    Crea una mappa assegnando un intero a ciascun protocollo nella lista fornita.

    Parameters:
    - protocol_list: list, lista di protocolli.

    Returns:
    - protocol_mapping: dict, mappa dei protocolli a interi.
    """
    protocol_mapping = {protocol: idx for idx, protocol in enumerate(protocol_list, start=1)}
    return protocol_mapping


def filter_and_save_csv(original_csv, filtered_csv, filtered_protocols, chunk_size=100000):
    """
    Filtra le righe del CSV originale mantenendo solo quelle con 'Protocol_Type' in filtered_protocols
    e salva il risultato in un nuovo CSV.

    Parameters:
    - original_csv: str, percorso al file CSV originale.
    - filtered_csv: str, percorso al file CSV filtrato da salvare.
    - filtered_protocols: set, insieme dei protocolli filtrati da mantenere.
    - chunk_size: int, numero di righe per ogni chunk (default: 100000).
    """
    try:
        # Rimuove il file di output se esiste già
        if os.path.exists(filtered_csv):
            os.remove(filtered_csv)

        # Inizializza la scrittura del nuovo CSV con l'intestazione
        first_chunk = True
        for chunk in pd.read_csv(original_csv, delimiter=';', chunksize=chunk_size):
            # Filtra le righe con Protocol_Type in filtered_protocols
            filtered_chunk = chunk[chunk['Protocol_Type'].isin(filtered_protocols)]

            # Reset degli indici del chunk filtrato
            filtered_chunk.reset_index(drop=True, inplace=True)

            # Scrive il chunk filtrato nel nuovo CSV
            filtered_chunk.to_csv(filtered_csv, mode='a', index=False, header=first_chunk, sep=';')
            if first_chunk:
                first_chunk = False
    except Exception as e:
        print(f"Si è verificato un errore durante il filtraggio e salvataggio del CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    csv_file_path = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/ton_iot_dataset.csv"
    filtered_csv = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/ton_iot_dataset_filtered.csv"
    chunk_size = 3000000
    threshold = 1000
    output_json_full = "json_full.json"
    output_json_filtered = "json_filtered.json"

    # 1. Conta le occorrenze di tutti i protocolli
    protocol_counts = create_protocol_counts(csv_file_path, chunk_size)

    # Converti il dizionario in una Serie di pandas per facilitare l'ordinamento
    protocol_counts_series = pd.Series(protocol_counts)

    # Ordina le occorrenze in maniera decrescente
    protocol_counts_sorted = protocol_counts_series.sort_values(ascending=False)

    # 2. Crea la mappa completa dei protocolli
    all_protocols_sorted = protocol_counts_sorted.index.tolist()
    protocol_mapping_full = create_mapping(all_protocols_sorted)

    # Stampa la mappa completa
    print("1. Mappa Completa dei Protocol_Type -> Intero:")
    print(protocol_mapping_full)
    print("\n" + "-" * 60 + "\n")

    # 3. Stampa il numero di occorrenze per protocollo in maniera decrescente
    print("2. Numero di Occorrenze per Protocol_Type (Decrescente):")
    for protocol, count in protocol_counts_sorted.items():
        print(f"{protocol}: {count}")
    print("\n" + "-" * 60 + "\n")

    # 4. Filtra i protocolli con occorrenze > threshold
    filtered_protocol_counts = protocol_counts_sorted[protocol_counts_sorted > threshold]

    if filtered_protocol_counts.empty:
        print(f"Nessun protocollo ha un numero di occorrenze maggiore di {threshold}.")
        sys.exit(0)

    # 5. Crea la mappa filtrata dei protocolli
    filtered_protocols_sorted = filtered_protocol_counts.index.tolist()
    protocol_mapping_filtered = create_mapping(filtered_protocols_sorted)

    # Stampa la mappa filtrata
    print(f"3. Mappa Filtrata dei Protocol_Type (Occorrenze > {threshold}) -> Intero:")
    print(protocol_mapping_filtered)
    print("\n" + "-" * 60 + "\n")

    # 6. Stampa il numero di occorrenze per protocollo filtrato in maniera decrescente
    print(f"4. Numero di Occorrenze per Protocol_Type (Occorrenze > {threshold}, Decrescente):")
    for protocol, count in filtered_protocol_counts.items():
        print(f"{protocol}: {count}")
    print("\n" + "-" * 60 + "\n")

    # 7. Salva le mappe in file JSON
    try:
        with open(output_json_full, 'w') as f_full:
            json.dump(protocol_mapping_full, f_full, indent=4)
        print(f"La mappa completa è stata salvata in '{output_json_full}'")
    except Exception as e:
        print(f"Errore nel salvataggio della mappa completa: {e}")

    try:
        with open(output_json_filtered, 'w') as f_filtered:
            json.dump(protocol_mapping_filtered, f_filtered, indent=4)
        print(f"La mappa filtrata è stata salvata in '{output_json_filtered}'")
    except Exception as e:
        print(f"Errore nel salvataggio della mappa filtrata: {e}")

    # 8. Filtra il CSV originale e salva il nuovo CSV filtrato
    print("\nInizio del filtraggio del CSV e salvataggio del nuovo CSV filtrato...")
    filter_and_save_csv(
        original_csv=csv_file_path,
        filtered_csv=filtered_csv,
        filtered_protocols=set(filtered_protocols_sorted),
        chunk_size=chunk_size
    )
    print(f"Il nuovo CSV filtrato è stato salvato in '{filtered_csv}'")