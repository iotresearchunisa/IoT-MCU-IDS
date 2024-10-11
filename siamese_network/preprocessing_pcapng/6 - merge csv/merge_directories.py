"""
### CSV Files Consolidator

This script consolidates multiple CSV files from a specified root directory (including its subdirectories) into a single CSV file. It performs the following tasks:

1. **Searches for CSV Files:** Recursively traverses the root directory to find all files with a `.csv` extension.
2. **Sorts Files Alphabetically:** Ensures that the CSV files are processed in alphabetical order.
3. **Merges CSV Files:** Reads each CSV file in chunks to handle large files efficiently and appends the data to a single output CSV file.
4. **Handles Headers Appropriately:** Writes the header only once to the output file and appends subsequent data without headers.

"""

import os
import pandas as pd
import sys

def find_csv_files(root_dir):
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                filepath = os.path.join(dirpath, filename)
                csv_files.append(filepath)
    csv_files.sort()
    return csv_files

def consolidate_csv_files(csv_files, output_file, chunksize=300000):
    header_saved = False

    for file in csv_files:
        print(f"Processing file: {file}")
        try:
            for chunk in pd.read_csv(file, delimiter=';', chunksize=chunksize, low_memory=False):
                # Reset the index of each chunk before appending it
                chunk = chunk.reset_index(drop=True)

                # Write the chunk to the output file
                if not header_saved:
                    chunk.to_csv(output_file, mode='w', index=False, sep=';', header=True)
                    header_saved = True
                else:
                    chunk.to_csv(output_file, mode='a', index=False, sep=';', header=False)
        except pd.errors.EmptyDataError:
            print(f"Warning: The file '{file}' is empty and has been skipped.")
        except Exception as e:
            print(f"Error processing file '{file}': {e}")

    print(f"\nAll files have been consolidated into {output_file}.")

def main():
    # Define the root directory containing CSV files
    root_dir = '/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_4'

    # Define the output CSV file path
    output_file = '/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/ton_iot_dataset.csv'

    # Find all CSV files in the root directory
    csv_files = find_csv_files(root_dir)

    if not csv_files:
        print(f"No CSV files found in the directory: {root_dir}")
        sys.exit(1)

    # Display the list of CSV files to be processed
    print("List of CSV files in alphabetical order:")
    for file in csv_files:
        print(file)

    # Consolidate the CSV files into a single output file
    consolidate_csv_files(csv_files, output_file)

if __name__ == '__main__':
    main()
