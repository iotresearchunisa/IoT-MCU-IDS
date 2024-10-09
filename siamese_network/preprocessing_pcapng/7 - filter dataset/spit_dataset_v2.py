import pandas as pd
import os
import sys


def process_csv_fixed_sample(input_file, output_file, chunk_size=100000, target_per_class=34601, delimiter=';'):
    # Definisci le classi originali e la mappatura per la rinominazione
    original_classes = ['benign', 'dos', 'password', 'scanning']
    rename_map = {
        'scanning': 'recon',
        'password': 'brute_force'
    }

    # Mappa le classi rinominate
    renamed_classes = [rename_map.get(cls, cls) for cls in original_classes]

    # Inizializza i contatori per ciascuna classe rinominata
    remaining_counts = {cls: target_per_class for cls in renamed_classes}

    # Flag per scrivere l'header solo una volta
    header_written = False

    try:
        # Leggi il CSV in chunk
        for chunk_number, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size, sep=delimiter), start=1):
            print(f"Elaborazione chunk {chunk_number}...")

            # Verifica la presenza della colonna 'type_attack'
            if 'type_attack' not in chunk.columns:
                print("Errore: La colonna 'type_attack' non è presente nel file CSV di input.")
                sys.exit(1)

            # Filtra le righe desiderate
            filtered_chunk = chunk[chunk['type_attack'].isin(original_classes)].copy()

            if filtered_chunk.empty:
                continue  # Passa al prossimo chunk se non ci sono righe filtrate

            # Rinomina i valori nella colonna 'type_attack'
            filtered_chunk['type_attack'] = filtered_chunk['type_attack'].replace(rename_map)

            # Lista per raccogliere le righe da scrivere nel file di output
            rows_to_write = []

            # Itera su ogni classe rinominata
            for cls in renamed_classes:
                if remaining_counts[cls] <= 0:
                    continue  # Se già raggiunto il target, salta

                # Seleziona le righe appartenenti alla classe corrente
                class_rows = filtered_chunk[filtered_chunk['type_attack'] == cls]

                if class_rows.empty:
                    continue  # Nessuna riga per questa classe nel chunk corrente

                # Determina quanti campioni prendere
                n_rows = min(len(class_rows), remaining_counts[cls])

                # Prendi le prime n_rows
                selected_rows = class_rows.head(n_rows)

                # Aggiungi le righe selezionate alla lista
                rows_to_write.append(selected_rows)

                # Aggiorna il contatore
                remaining_counts[cls] -= n_rows

                print(f"  Classe '{cls}': Aggiunte {n_rows} righe (Rimanenti: {remaining_counts[cls]})")

                # Se tutti i target sono raggiunti, termina l'elaborazione
                if all(count <= 0 for count in remaining_counts.values()):
                    break

            if rows_to_write:
                # Combina tutte le righe da scrivere in questo chunk
                output_subset = pd.concat(rows_to_write)

                # Scrivi nel file CSV di output
                if not header_written:
                    output_subset.to_csv(output_file, index=False, mode='w', sep=delimiter)
                    header_written = True
                else:
                    output_subset.to_csv(output_file, index=False, mode='a', header=False, sep=delimiter)

            # Verifica se tutti i target sono stati raggiunti
            if all(count <= 0 for count in remaining_counts.values()):
                print("Tutti i target per le classi sono stati raggiunti. Terminazione dell'elaborazione.")
                break

        # Informazioni finali
        print("\nProcesso completato.")
        print(f"Il file filtrato è stato salvato in '{output_file}'.")
        print("Riepilogo delle classi:")
        for cls in renamed_classes:
            selected = target_per_class - remaining_counts[cls]
            print(f"  {cls}: {selected} righe")

    except pd.errors.EmptyDataError:
        print("Errore: Il file CSV di input è vuoto.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Errore nel parsing del CSV: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Errore inaspettato: {e}")
        sys.exit(1)


if __name__ == "__main__":
    input_path = "/mnt/FE9090E39090A3A5/Tesi/TON_IoT/ton_iot_dataset_filtered.csv"
    output_pat = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/siamese_network/datasets/TON_IoT/dataset.csv"
    chunk_size = 300000

    # Verifica che il file di input esista
    if not os.path.isfile(input_path):
        print(f"Errore: Il file di input '{input_path}' non esiste.")
        sys.exit(1)

    # Chiama la funzione di elaborazione
    process_csv_fixed_sample(input_path, output_pat, chunk_size)
