import os
import pandas as pd

def check_column_types_consistency(input_root):
    # Dizionario per memorizzare i tipi di dati del primo file CSV
    reference_column_types = None
    inconsistent_files = []
    inconsistent_columns = {}

    # Attraversa tutte le directory e i file nella root di input
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                # Leggi il CSV
                df = pd.read_csv(input_file_path, delimiter=';', low_memory=False)

                # Ottieni i tipi di dati delle colonne
                current_column_types = df.dtypes.to_dict()

                if reference_column_types is None:
                    # Memorizza il primo file come riferimento per il controllo
                    reference_column_types = current_column_types
                    print(f"Reference column types set from: {input_file_path}")
                else:
                    # Confronta i tipi di dati con il riferimento
                    for column, dtype in current_column_types.items():
                        if column not in reference_column_types:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' found in {input_file_path}, but not in reference")
                        elif reference_column_types[column] != dtype:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' in {input_file_path} has type {dtype}, expected {reference_column_types[column]}")

                    # Trova le colonne mancanti nel file corrente rispetto al riferimento
                    for column in reference_column_types:
                        if column not in current_column_types:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' missing in {input_file_path}")

                    # Aggiungi il file alla lista dei file inconsistenti se ci sono differenze
                    if any(column in inconsistent_columns for column in current_column_types):
                        inconsistent_files.append(input_file_path)

    # Report finale
    if inconsistent_columns:
        print("\nInconsistent columns found:")
        for column, messages in inconsistent_columns.items():
            print(f"\nColumn: {column}")
            for message in messages:
                print(f"  - {message}")
    else:
        print("\nAll files have consistent column types.")

    if inconsistent_files:
        print("\nFiles with inconsistencies:")
        for file in inconsistent_files:
            print(f"- {file}")
    else:
        print("\nAll files are consistent.")

if __name__ == "__main__":
    # Specifica la root della directory di input
    input_root = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_4"

    # Verifica la consistenza dei tipi di colonne in tutti i file CSV
    check_column_types_consistency(input_root)
