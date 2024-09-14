import os
import pandas as pd

def analyze_csv_files(root_directory):
    column_status = {}

    # Funzione per verificare se una colonna è completamente vuota o contiene solo zeri o NaN
    def is_empty_or_zero(column):
        # Escludiamo solo le colonne che contengono esclusivamente 0 o NaN o False
        return (column.isna() | (column == 0) | (column == "0.0") | (column == "False") | (column == False)).all()
    
    # Funzione per rilevare e stampare righe malformate
    def check_malformed_rows(file_path):
        with open(file_path, 'r') as file:
            # Legge solo la prima riga per determinare il numero di colonne attese
            first_line = file.readline()
            expected_columns = len(first_line.split(';'))

            malformed_rows = []
            for i, line in enumerate(file, start=2):  # start=2 per contare correttamente le righe
                num_columns = len(line.split(';'))
                if num_columns != expected_columns:
                    malformed_rows.append((i, num_columns, expected_columns, line.strip()))

            if malformed_rows:
                print(f"Righe malformate trovate nel file {file_path}:")
                for row in malformed_rows:
                    print(f"  Riga {row[0]} ha {row[1]} colonne, attese {row[2]}. Contenuto: {row[3]}")

    # Cerca i file CSV all'interno delle sottodirectory
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                print(f"Analizzando il file: {file_path}")
                try:
                    # Controllo righe malformate prima di caricare il CSV
                    check_malformed_rows(file_path)

                    df = pd.read_csv(file_path, delimiter=';', low_memory=False)

                    # Verifica se ci sono colonne vuote o uguali a zero
                    for col in df.columns:
                        if col not in column_status:
                            column_status[col] = True  # Assume che la colonna sia vuota o zero in tutti i file
                        if not is_empty_or_zero(df[col]):
                            column_status[col] = False  # Se la colonna non è vuota/zero in questo file, segna come False
                except Exception as e:
                    print(f"Errore durante la lettura del file {file_path}: {e}")

    # Stampa solo le colonne che sono vuote o uguali a zero in tutti i CSV
    columns_to_report = [col for col, status in column_status.items() if status]

    if columns_to_report:
        print("Le seguenti colonne sono vuote o contengono solo zeri o NaN in tutti i CSV analizzati:")
        for col in columns_to_report:
            print(col)
    else:
        print("Non ci sono colonne vuote o contenenti solo zeri o NaN in tutti i CSV analizzati.")

# Usa la funzione
root_directory = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned"
analyze_csv_files(root_directory)
