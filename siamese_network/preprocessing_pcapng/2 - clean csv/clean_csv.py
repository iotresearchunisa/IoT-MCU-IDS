"""

This Python script is designed to clean CSV files generated from network traffic analysis, particularly those derived from PCAPNG files.
The main functionalities of the program include:

1. **Protocol Filtering**: Removes rows where the 'Protocol_Type' column contains unwanted protocols.
2. **Boolean Conversion**: Converts boolean values "True" and "False" to integers 1 and 0, respectively.
3. **Handling Missing Values**: Replaces empty cells with 0 to ensure data consistency.
4. **Rate Filtering**: Eliminates rows where the 'rate' column has a value of 0, which may indicate inactive or irrelevant packets.
5. **MQTT Packet Classification**: Marks specific MQTT messages (Ping, Keep-Alive, and PUBLISH) as 'benign' to differentiate them from potential threats.
6. **TCP Packet Classification**: Identifies and marks TCP packets that immediately follow the specified MQTT messages as 'benign', assuming they are part of the benign MQTT communication.

"""

import pandas as pd
import os
import argparse
import glob
import sys


def clean_csv(input_csv, output_csv):
    try:
        # Initialize variables
        chunksize = 300000  # Adjust this number based on your system's memory capacity
        header_written = False  # Flag to ensure header is written only once

        # Protocols to remove
        protocols_to_remove = [
            '802.11', 'LLC', 'IPv4', 'IPv6', 'VRRP', 'RPL', 'RSVP-E2EI', 'X.25', 'IP',
            'DLR', 'ISO', 'MIPv6', 'CLNP', 'IGRP', 'RSVP', 'UDP-Lite', 'ICMPv6'
        ]

        # MQTT flags column
        mqtt_flags = 'MQTT_HeaderFlags'  # Replace with the correct column in your dataset

        # Read and process the CSV in chunks
        for chunk in pd.read_csv(input_csv, delimiter=';', low_memory=False, chunksize=chunksize):
            # Remove rows where 'Protocol_Type' is in protocols_to_remove
            chunk = chunk[~chunk['Protocol_Type'].isin(protocols_to_remove)]

            # Convert boolean values True to 1 and False to 0
            chunk.replace({True: 1, False: 0, "True": 1, "False": 0}, inplace=True)

            # Assign 0 to empty cells
            chunk.fillna(0, inplace=True)

            # Remove rows where the 'rate' column has a value of 0
            if 'rate' in chunk.columns:
                chunk = chunk[chunk['rate'] != 0]

            # Reset indices after row removal
            chunk.reset_index(drop=True, inplace=True)

            # Ensure 'type_attack' column exists and is of string type
            if 'type_attack' in chunk.columns:
                chunk['type_attack'] = chunk['type_attack'].astype(str)
            else:
                # If 'type_attack' does not exist, create it as a string column
                chunk['type_attack'] = ''

            # Convert hexadecimal string values to integers in mqtt_flags column
            if mqtt_flags in chunk.columns:
                chunk[mqtt_flags] = chunk[mqtt_flags].apply(
                    lambda x: int(x, 16) if isinstance(x, str) and '0x' in x else x
                )

                # Assign "benign" to packets with specific MQTT flags
                chunk['type_attack'] = chunk.apply(
                    lambda row: 'benign' if row[mqtt_flags] in [0xC0, 0xD0, 0x30] else row['type_attack'],
                    axis=1
                )

            # Find MQTT packet indices
            mqtt_indices = chunk.index[chunk["Protocol_Type"] == "MQTT"].tolist()

            # For each MQTT packet, mark the next two TCP packets as 'benign'
            for idx in mqtt_indices:
                next_indices = [idx + 1, idx + 2]
                for next_idx in next_indices:
                    if next_idx < len(chunk) and chunk.at[next_idx, 'Protocol_Type'] == 'TCP':
                        chunk.at[next_idx, 'type_attack'] = 'benign'

            # Write the processed chunk to the output CSV file
            if not header_written:
                chunk.to_csv(output_csv, sep=';', index=False, mode='w')
                header_written = True
            else:
                chunk.to_csv(output_csv, sep=';', index=False, mode='a', header=False)

        print(f"Cleaned CSV saved to {output_csv}")

    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="Clean CSV files by removing specific 'Protocol_Type' rows and converting True/False values.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Parameter for the input files or directory
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        help="Path to the CSV file(s) or directory containing CSV files."
    )

    # Parameter for the output file or directory
    parser.add_argument(
        '-o', '--output',
        help="Path to the output CSV file or directory where the results will be saved."
    )

    # Optional parameter to specify custom output file names
    parser.add_argument(
        '-n', '--names',
        nargs='*',
        help="Optional custom output CSV file names (if multiple files are given)."
    )

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
                # If custom names are provided, use them; otherwise, use default cleaned name
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
