import os
import pandas as pd

# Imposta la path di root
root_path = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_2'

# Lista delle colonne da cancellare
colonne_da_cancellare = [
    'MQTT_ConAck_Reserved',
    'MQTT_ConAck_SP',
    'MQTT_ConAck_Val',
    'MQTT_Username',
    'MQTT_Password',
    'MQTT_Sub_QoS',
    'MQTT_SubAck_QoS',
    'MQTT_WillMsg',
    'MQTT_WillMsg_Length',
    'MQTT_WillTopic',
    'MQTT_WillTopic_Length'
]

# Percorri tutte le sottodirectory e i file
for subdir, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)
            try:
                # Leggi il CSV con delimitatore ';'
                df = pd.read_csv(file_path, delimiter=';', low_memory=False)
                # Cancella le colonne specificate
                df.drop(columns=colonne_da_cancellare, inplace=True, errors='ignore')
                # Salva di nuovo il CSV con delimitatore ';'
                df.to_csv(file_path, index=False, sep=';')
                print(f"Colonne {colonne_da_cancellare} rimosse da {file_path}")
            except Exception as e:
                print(f"Errore nel processare {file_path}: {e}")
