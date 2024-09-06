import os
import argparse
import pandas as pd
import sys


def merge_csv(input_files, output_file):
    """
    Function to merge multiple CSV files into one.

    :param input_files: List of paths to the input CSV files.
    :param output_file: Path to the output merged CSV file.
    """
    try:
        # Read all CSV files and merge them
        dataframes = []
        for file in input_files:
            df = pd.read_csv(file, delimiter=';')
            dataframes.append(df)

        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)

        # Save the merged dataframe to a new CSV file
        merged_df.to_csv(output_file, index=False, sep=';')
        print(f"Merged CSV saved to {output_file}")

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    # Use argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Parameter for the input files or directory
    parser.add_argument('-i', '--input', nargs='+', help="Path to the CSV file(s) or directory containing CSV files.")

    # Parameter for the output file or directory
    parser.add_argument('-o', '--output', help="Path to the output CSV file or directory where the result will be saved.")

    # Optional parameter to specify custom output file name
    parser.add_argument('-n', '--name', help="Optional custom output CSV file name.")

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
        csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in directory: {input_dir}")
            sys.exit(1)

        # Ensure that there are at least two CSV files in the directory
        if len(csv_files) < 2:
            print(f"Error: The directory must contain at least two CSV files for merging.")
            sys.exit(1)

        # If a name is provided, use it, otherwise use default name
        output_filename = args.name if args.name else 'merged.csv'
        output_file = os.path.join(args.output, output_filename)

        merge_csv(csv_files, output_file)

    # Case 2: If multiple files are given, ensure at least two files
    else:
        if len(args.input) < 2:
            print("Error: You must provide at least two CSV files for merging.")
            sys.exit(1)

        # If a name is provided, use it, otherwise use default name
        output_filename = args.name if args.name else 'merged.csv'
        output_file = os.path.join(args.output, output_filename)

        merge_csv(args.input, output_file)


if __name__ == "__main__":
    main()
