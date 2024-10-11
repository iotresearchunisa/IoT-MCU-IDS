"""
This Python script processes and transforms CSV files by converting specified columns to `float64` data type,
including handling hexadecimal values in certain columns. The script operates recursively over all CSV files
in the input directory and saves the transformed files in the output directory.

### Main functionalities:
1. **Conversion of specific columns to `float64`**: Columns such as 'Time_To_Leave', 'Header_Length', and
   others are converted to `float64` if they are not already in that format.

2. **Conversion of hexadecimal values**: Columns like 'MQTT_ConFlags' and 'MQTT_ConAck_Flags', which contain
   hexadecimal values (e.g., '0x00'), are converted to `float64`.

3. **Removal of a specified column**: The script removes the column 'MQTT_Proto_Name' if it exists in the dataset.

4. **Recursive processing**: The script processes all CSV files in the input directory and its subdirectories,
   preserving the directory structure in the output directory.

### Usage:
- **Input**: Specify the path of the input directory containing the CSV files.
- **Output**: Specify the path of the output directory where the transformed CSV files will be saved.
"""

import os
import pandas as pd


def hex_to_float(cell):
    try:
        if isinstance(cell, str) and cell.startswith('0x'):
            return float(int(cell, 16))
        return float(cell)
    except ValueError:
        return pd.NA


def process_csv(input_file_path, output_file_path):
    # List of columns to convert to float64
    columns_to_convert = [
        'Time_To_Leave', 'Header_Length', 'Packet_Fragments', 'rate',
        'TCP_Flag_FIN', 'TCP_Flag_SYN', 'TCP_Flag_RST', 'TCP_Flag_PSH', 'TCP_Flag_ACK',
        'TCP_Flag_ECE', 'TCP_Flag_CWR', 'Packet_Length', 'Packet_Fragments',
        'TCP_Length', 'MQTT_CleanSession', 'MQTT_QoS', 'MQTT_Reserved', 'MQTT_Retain',
        'MQTT_WillFlag', 'MQTT_DupFlag', 'MQTT_HeaderFlags', 'MQTT_KeepAlive',
        'MQTT_Length', 'MQTT_MessageType', 'MQTT_Proto_Length', 'MQTT_Conflag_QoS',
        'MQTT_Conflag_Retain', 'MQTT_Version'
    ]

    # List of hexadecimal columns to convert to float64
    hex_columns = ['MQTT_ConFlags', 'MQTT_ConAck_Flags']

    # Columns to convert to string
    string_columns = ['type_attack', 'Protocol_Type']

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Define the chunksize
    chunksize = 300000

    # Initialize the write_header flag
    write_header = True

    # Open the output file for writing
    with open(output_file_path, 'w', newline='', encoding='utf-8') as f_output:
        # Read the CSV file in chunks
        for chunk in pd.read_csv(input_file_path, delimiter=';', chunksize=chunksize, low_memory=False):
            # Rimuove la colonna 'MQTT_Proto_Name' se esiste
            if 'MQTT_Proto_Name' in chunk.columns:
                chunk.drop(columns=['MQTT_Proto_Name'], inplace=True)

            # Converti le colonne esadecimali a float64
            for col in hex_columns:
                if col in chunk.columns:
                    chunk[col] = chunk[col].apply(hex_to_float)

            # Converti le colonne specificate a float64, se esistono e non sono già float64
            for col in columns_to_convert:
                if col in chunk.columns and chunk[col].dtype != 'float64':
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype('float64')

            # Converti le colonne specificate a string, indipendentemente dal loro tipo attuale
            for col in string_columns:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(str)

            # Scrivi il chunk nel file CSV di output
            chunk.to_csv(f_output, index=False, sep=';', header=write_header, mode='a')
            write_header = False  # Scrivi l'intestazione solo per il primo chunk

    print(f"Processed: {input_file_path}")


def process_all_csvs(input_root, output_root):
    # Traversata di tutte le directory e file in input_root
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                # Crea il percorso del file di output corrispondente
                relative_path = os.path.relpath(input_file_path, input_root)
                output_file_path = os.path.join(output_root, relative_path)

                # Processa il file CSV e salva l'output
                process_csv(input_file_path, output_file_path)


if __name__ == "__main__":
    input_root = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_3"
    output_root = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_4"

    process_all_csvs(input_root, output_root)