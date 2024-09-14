import os
import pandas as pd

root_path = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_2'

# List of columns to remove from CSV files
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

# Traverse through all subdirectories and files
for subdir, dirs, files in os.walk(root_path):
    for file in files:
        # Only process .csv files
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)  # Full path to the current file
            try:
                df = pd.read_csv(file_path, delimiter=';', low_memory=False)

                # Remove the specified columns, ignoring errors if a column doesn't exist
                df.drop(columns=colonne_da_cancellare, inplace=True, errors='ignore')

                df.to_csv(file_path, index=False, sep=';')

                print(f"Columns {colonne_da_cancellare} removed from {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")