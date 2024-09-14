"""
### CSV Files Merger

This script merges multiple CSV files into a single CSV file. It performs the following tasks:

1. **Merge CSV Files:** Reads multiple CSV files of the same directory and merges them into one dataframe.
2. **Command-Line Interface:** Uses `argparse` to handle command-line arguments, allowing users to specify input files or directories and an output file.
3. **Handles Large Data:** Efficiently processes CSV files using pandas.
4. **Custom Output Filename:** Optionally allows users to specify a custom name for the merged output CSV file.

"""

import os
import argparse
import pandas as pd
import sys


def merge_csv(input_files, output_file):
    """
    Merge multiple CSV files into a single output file.

    Args:
        input_files (list): List of paths to the input CSV files.
        output_file (str): Path to the output merged CSV file.

    Returns:
        None
    """
    try:
        # List to store the dataframes
        dataframes = []

        # Read each CSV file and store its dataframe
        for file in input_files:
            print(f"Reading file: {file}")
            df = pd.read_csv(file, delimiter=';', low_memory=False)
            dataframes.append(df)

        # Concatenate all dataframes into one
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Save the merged dataframe to a new CSV file
        merged_df.to_csv(output_file, index=False, sep=';')
        print(f"Merged CSV saved to {output_file}")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def main():
    """
    Main function to handle the CSV merging process based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Parameter for the input files or directory
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help="Path to the CSV file(s) or directory containing CSV files.")

    # Parameter for the output file or directory
    parser.add_argument('-o', '--output', required=True,
                        help="Path to the output CSV file or directory where the result will be saved.")

    # Optional parameter to specify a custom output file name
    parser.add_argument('-n', '--name',
                        help="Optional custom output CSV file name.")

    args = parser.parse_args()

    # Case 1: If input is a directory, process all CSV files in that directory
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        input_dir = args.input[0]
        csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.csv')]

        if not csv_files:
            print(f"No CSV files found in directory: {input_dir}")
            sys.exit(1)

        if len(csv_files) < 2:
            print(f"Error: The directory must contain at least two CSV files for merging.")
            sys.exit(1)

        # Use the provided name or default to 'merged.csv'
        output_filename = args.name if args.name else 'merged.csv'
        output_file = str(os.path.join(args.output, output_filename))

        merge_csv(csv_files, output_file)

    # Case 2: If multiple input files are given, ensure there are at least two
    else:
        if len(args.input) < 2:
            print("Error: You must provide at least two CSV files for merging.")
            sys.exit(1)

        output_filename = args.name if args.name else 'merged.csv'
        output_file = str(os.path.join(args.output, output_filename))

        merge_csv(args.input, output_file)


if __name__ == "__main__":
    main()
