import os
import pandas as pd

def hex_to_float(cell):
    """Converts a hexadecimal string (e.g. '0x00') to a float."""
    try:
        if isinstance(cell, str) and cell.startswith('0x'):
            return float(int(cell, 16))
        return float(cell)
    except ValueError:
        return pd.NA

def process_csv(input_file_path, output_file_path):
    # Carica il file CSV
    df = pd.read_csv(input_file_path, delimiter=';', low_memory=False)

    # Liste delle colonne da convertire in float64
    columns_to_convert = [
        'Time_To_Leave', 'Header_Length', 'Packet_Fragments', 'rate',
        'TCP_Flag_FIN', 'TCP_Flag_SYN', 'TCP_Flag_RST', 'TCP_Flag_PSH', 'TCP_Flag_ACK',
        'TCP_Flag_ECE', 'TCP_Flag_CWR', 'Packet_Length', 'Packet_Fragments',
        'TCP_Length', 'MQTT_CleanSession', 'MQTT_QoS', 'MQTT_Reserved', 'MQTT_Retain',
        'MQTT_WillFlag', 'MQTT_DupFlag', 'MQTT_HeaderFlags', 'MQTT_KeepAlive',
        'MQTT_Length', 'MQTT_MessageType', 'MQTT_Proto_Length', 'MQTT_Conflag_QoS',
        'MQTT_Conflag_Retain', 'MQTT_Version'
    ]

    # Conversione delle colonne esadecimali a float64
    hex_columns = ['MQTT_ConFlags', 'MQTT_ConAck_Flags']

    # Elimina la colonna 'MQTT_Proto_Name' se esiste
    if 'MQTT_Proto_Name' in df.columns:
        df.drop(columns=['MQTT_Proto_Name'], inplace=True)

    # Converte le colonne esadecimali in float64
    for col in hex_columns:
        if col in df.columns:
            df[col] = df[col].apply(hex_to_float)

    # Converte le colonne specificate in float64, se esistono
    for col in columns_to_convert:
        if col in df.columns and df[col].dtype != 'float64':
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

    # Crea la directory di output se non esiste
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Salva il CSV trasformato
    df.to_csv(output_file_path, index=False, sep=';')

    print(f"Processed: {input_file_path}")


def process_all_csvs(input_root, output_root):
    # Walk through all directories and files in the input_root
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                # Create the corresponding output file path
                relative_path = os.path.relpath(input_file_path, input_root)
                output_file_path = os.path.join(output_root, relative_path)

                # Process the CSV file and save the output
                process_csv(input_file_path, output_file_path)


if __name__ == "__main__":
    # Specifica la root della directory di input e output
    input_root = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_3"
    output_root = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_4"

    # Esegui il processo su tutti i file CSV
    process_all_csvs(input_root, output_root)
