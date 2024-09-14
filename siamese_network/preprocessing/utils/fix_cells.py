import pandas as pd


# Funzione per estrarre il secondo valore se ci sono due valori separati da ',' e ignorare i valori vuoti
def extract_second_value(cell):
    if pd.isna(cell) or cell == '':
        return cell  # Ignora i valori vuoti
    if ',' in cell:
        return cell.split(',')[1]
    return cell


# Funzione aggiornata per leggere un CSV da un input file, ignorare i valori vuoti e salvare in un output file
def process_csv(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, delimiter=';', low_memory=False)

    # Conta le righe che contengono almeno una cella con due valori separati da ','
    rows_with_multiple_values = (df.astype(str).apply(lambda row: row.str.contains(',')).any(axis=1)).sum()

    # Applica la funzione per estrarre il secondo valore, ignorando i valori vuoti
    df_transformed = df.astype(str).apply(lambda col: col.apply(extract_second_value))

    # Salva il DataFrame trasformato nel file di output
    df_transformed.to_csv(output_file_path, index=False, sep=';')

    print(f"Il CSV trasformato è stato salvato come {output_file_path}")
    print(f"Numero di righe con più di un valore separato da virgola: {rows_with_multiple_values}")

    return rows_with_multiple_values


input_file = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv/attacks/dos/ACK_Fragmentation/flood.csv'
output_file = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned/attacks/dos/ACK_Fragmentation/flood.csv'

process_csv(input_file, output_file)
