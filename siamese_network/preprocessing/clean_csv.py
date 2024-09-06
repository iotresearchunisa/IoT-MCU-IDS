import csv
import numpy as np
import os
import argparse
import glob
import sys

def clean_csv(input_csv, output_csv):
    """
    Function to remove the first row after the header and rows where 'Protocol_Type' is '802.11', 'LLC', 'IPv4', 'IPv6', etc.
    Also converts "True" to 1, "False" to 0, and empty cells to NaN.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output cleaned CSV file.
    """
    try:
        # Open the input CSV file for reading
        with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=';')

            # Read the header and the rest of the file
            header = next(reader)  # First row (header)
            rows = list(reader)  # All rows

        # Remove the first row after the header
        if rows:
            rows.pop(0)

        # Filter rows where 'Protocol_Type' is not in the list
        protocol_index = header.index('Protocol_Type')
        filtered_rows = [row for row in rows if row[protocol_index] not in [
            '802.11', 'LLC', 'IPv4', 'IPv6', 'VRRP', 'RPL', 'RSVP-E2EI', 'X.25', 'IP',
            'DLR', 'ISO', 'MIPv6', 'CLNP', 'IGRP', 'RSVP', 'UDP-Lite', 'ICMPv6']]

        # Replace "True" with 1, "False" with 0, and empty cells with np.nan
        for row in filtered_rows:
            for i in range(len(row)):
                if row[i] == "True":
                    row[i] = 1  # Keep as integer
                elif row[i] == "False":
                    row[i] = 0  # Keep as integer
                elif row[i] == "":
                    row[i] = np.nan  # Use np.nan for missing values

        # Write the cleaned rows to the output CSV file
        with open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=';')
            # Write the header back to the file
            writer.writerow(header)
            # Write the filtered and updated rows
            writer.writerows(filtered_rows)

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
