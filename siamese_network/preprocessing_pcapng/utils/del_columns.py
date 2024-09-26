"""

This script processes all CSV files located in a specified root directory and its subdirectories.
For each CSV file, it:

1. Removes specific columns based on a predefined list.
2. Renames the headers of certain columns by removing '.1' from 'MQTT_QoS.1' and 'MQTT_Retain.1'.
3. Saves the processed CSV files into a new output directory, maintaining the original folder structure.

It handles potential errors gracefully and prints out progress information for each file processed

"""

import os
import pandas as pd


root_path = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_2'
output_root_path = '/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_3'

# List of columns to be removed from each CSV file
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

# Traverse through all subdirectories and files in the root_path
for subdir, dirs, files in os.walk(root_path):
    for file in files:
        # Only process files with .csv extension
        if file.endswith('.csv'):
            file_path = os.path.join(subdir, file)  # Full path to the current CSV file

            # Determine the relative path of the file to maintain folder structure in the output directory
            relative_path = os.path.relpath(file_path, root_path)
            output_file_path = os.path.join(output_root_path, relative_path)

            # Create the output directory structure if it doesn't exist
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            try:
                print(f"Processing file: {file_path}")

                df = pd.read_csv(file_path, delimiter=';', low_memory=False)

                # Remove the specified columns; if a column doesn't exist, ignore the error
                df.drop(columns=colonne_da_cancellare, inplace=True, errors='ignore')

                # Modify headers by removing '.1' from 'MQTT_QoS.1' and 'MQTT_Retain.1'
                df.columns = df.columns.str.replace('MQTT_QoS.1', 'MQTT_Conflag_QoS', regex=False)
                df.columns = df.columns.str.replace('MQTT_Retain.1', 'MQTT_Conflag_Retain', regex=False)

                # Save the modified CSV file in the new output directory
                df.to_csv(output_file_path, index=False, sep=';')

                print(f"Columns {colonne_da_cancellare} removed and headers updated in {output_file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
