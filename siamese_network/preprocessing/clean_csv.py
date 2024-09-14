import pandas as pd
import os
import argparse
import glob
import sys


def clean_csv(input_csv, output_csv):
    """
    Function to remove rows where 'Protocol_Type' is in the list of unwanted protocols.
    Also converts "True" to 1, "False" to 0, and empty cells to NaN. Additionally, removes rows where 'rate' is 0 and marks
    MQTT Ping, Keep-Alive, and PUBLISH messages as benign, as well as TCP packets immediately following those messages.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output cleaned CSV file.
    """
    try:
        # Load the input CSV into a DataFrame
        df = pd.read_csv(input_csv, delimiter=';', low_memory=False)

        # Protocolli da rimuovere
        protocols_to_remove = ['802.11', 'LLC', 'IPv4', 'IPv6', 'VRRP', 'RPL', 'RSVP-E2EI', 'X.25', 'IP',
                               'DLR', 'ISO', 'MIPv6', 'CLNP', 'IGRP', 'RSVP', 'UDP-Lite', 'ICMPv6']

        # Elimina le righe dove 'Protocol_Type' è in protocols_to_remove
        df = df[~df['Protocol_Type'].isin(protocols_to_remove)]

        # Converti i valori booleani True in 1 e False in 0
        df.replace({True: 1, False: 0, "True": 1, "False": 0}, inplace=True)

        # Assegna 0 alle celle vuote
        df.fillna(0, inplace=True)

        # Elimina le righe dove la colonna 'rate' ha valore 0
        if 'rate' in df.columns:
            df = df[df['rate'] != 0]

        # Reimposta gli indici dopo l'eliminazione delle righe
        df.reset_index(drop=True, inplace=True)

        # Identifica pacchetti MQTT Ping, Keep-Alive e PUBLISH e assegna il valore 'benign'
        mqtt_flags = 'MQTT_HeaderFlags'  # Sostituisci con la colonna corretta nel dataset
        if mqtt_flags in df.columns:
            # Converte i valori della colonna in numeri interi se sono stringhe esadecimali
            df[mqtt_flags] = df[mqtt_flags].apply(lambda x: int(x, 16) if isinstance(x, str) and '0x' in x else x)

            # Assegna "benign" ai pacchetti con specifici flag MQTT
            df['type_attack'] = df.apply(
                lambda row: 'benign' if row[mqtt_flags] in [0xC0, 0xD0, 0x30] else row['type_attack'],
                axis=1
            )

        # Trova i pacchetti TCP successivi a pacchetti MQTT (Ping, Keep-Alive, PUBLISH)
        mqtt_indices = df.index[df["Protocol_Type"] == "MQTT"].tolist()
        for idx in mqtt_indices:
            # Prendi i prossimi due pacchetti dopo il pacchetto MQTT
            next_two_rows = df.iloc[idx + 1:idx + 3]

            # Verifica se esiste un primo pacchetto
            if len(next_two_rows) > 0:
                # Se il primo pacchetto è TCP, classificalo come benign
                if next_two_rows.iloc[0]['Protocol_Type'] == 'TCP':
                    df.loc[next_two_rows.index[0], 'type_attack'] = 'benign'

            # Verifica se esiste un secondo pacchetto
            if len(next_two_rows) > 1:
                # Se il secondo pacchetto è TCP, classificalo come benign
                if next_two_rows.iloc[1]['Protocol_Type'] == 'TCP':
                    df.loc[next_two_rows.index[1], 'type_attack'] = 'benign'

        # Salva il DataFrame pulito nel file CSV di output
        df.to_csv(output_csv, sep=';', index=False)

        print(f"Cleaned CSV saved to {output_csv}")

    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Clean CSV files by removing specific 'Protocol_Type' rows and converting True/False values.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Parameter for the input files or directory
    parser.add_argument('-i', '--input', nargs='+', help="Path to the CSV file(s) or directory containing CSV files.")

    # Parameter for the output file or directory
    parser.add_argument('-o', '--output', help="Path to the output CSV file or directory where the results will be saved.")

    # Optional parameter to specify custom output file names
    parser.add_argument('-n', '--names', nargs='*', help="Optional custom output CSV file names (if multiple files are given).")

    args = parser.parse_args()

    # Check if input and output parameters are provided
    if not args.input:
        print("Error: The -i or --input flag is required to specify the CSV file(s) or directory.")
        parser.print_help()
        sys.exit(1)

    if not args.output:
        print("Error: The -o or --output flag is required to specify the output CSV file or directory.")
        parser.print_help()
        sys.exit(1)

    # Case 1: If the input is a directory, process all CSV files in the directory
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        input_dir = args.input[0]
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        if not csv_files:
            print(f"No CSV files found in directory: {input_dir}")
            sys.exit(1)

        if not os.path.isdir(args.output):
            print("Error: The output must be a directory when processing multiple files from a directory.")
            sys.exit(1)

        for csv_file in csv_files:
            csv_filename = os.path.basename(csv_file)
            output_file = os.path.join(args.output, csv_filename.replace('.csv', '_cleaned.csv'))
            clean_csv(csv_file, output_file)

    # Case 2: If multiple files or a single file is given, process them
    else:
        if not os.path.isdir(args.output):
            print("Error: The output must be a directory when processing multiple files.")
            sys.exit(1)

        # If custom names are provided, they must match the number of input files
        if args.names and len(args.names) != len(args.input):
            print("Error: The number of custom names must match the number of input files.")
            sys.exit(1)

        for i, csv_file in enumerate(args.input):
            if os.path.isfile(csv_file):
                # If custom names are provided, use them, otherwise use default cleaned name
                if args.names:
                    output_filename = args.names[i]
                    if not output_filename.endswith('.csv'):
                        output_filename += '.csv'
                else:
                    csv_filename = os.path.basename(csv_file)
                    output_filename = csv_filename.replace('.csv', '_cleaned.csv')

                output_file = os.path.join(args.output, output_filename)
                clean_csv(csv_file, output_file)
            else:
                print(f"Error: File '{csv_file}' does not exist.")
                sys.exit(1)


if __name__ == "__main__":
    main()
