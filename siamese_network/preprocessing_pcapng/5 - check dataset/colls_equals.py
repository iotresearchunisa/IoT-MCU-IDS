"""
This Python script analyzes CSV files within a specified directory, with the following key functionalities:

1. **Check for Malformed Rows**:
   - The script examines each CSV file to detect rows where the number of columns differs from the expected number (determined by the first row).
   - Any malformed rows (i.e., rows with missing or extra columns) are reported along with the row number and content for easier debugging.

2. **Identify Empty or Zero-Filled Columns**:
   - It identifies columns across all CSV files that are completely empty, filled with zeros (`0`), or contain only `NaN` or `False` values.
   - The script checks each CSV file and flags columns that are empty or zero-filled consistently across all the files analyzed.

3. **Generate a Final Report**:
   - After processing all CSV files, the script generates a report listing columns that are entirely empty or contain only zeros, `NaN`, or `False` values across all CSV files.

4. **Handles Errors Gracefully**:
   - The script includes error handling to gracefully skip problematic files and report any issues encountered during file processing.

5. **Processes CSV Files in Chunks**:
   - The script reads and processes large CSV files in chunks to handle memory efficiently and work with large datasets.
"""

import os
import pandas as pd

def analyze_csv_files(root_directory):
    column_status = {}

    # Function to check if a column is completely empty or contains only zeros, NaN, or False
    def is_empty_or_zero(column):
        # Check if the entire column contains only NaNs, 0, "0.0", or False
        return (column.isna() | (column == 0) | (column == "0.0") | (column == "False") | (column == False)).all()

    # Function to detect and print malformed rows (rows with incorrect number of columns)
    def check_malformed_rows(file_path, expected_columns):
        malformed_rows = []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file, start=1):
                num_columns = len(line.rstrip('\n').split(';'))
                if num_columns != expected_columns:
                    malformed_rows.append((i, num_columns, expected_columns, line.strip()))
        if malformed_rows:
            print(f"\nMalformed rows found in file {file_path}:")
            for row in malformed_rows:
                print(f"  Row {row[0]} has {row[1]} columns, expected {row[2]}. Content: {row[3]}")

    # Search for CSV files within the subdirectories
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = str(os.path.join(subdir, file))
                print(f"\nAnalyzing file: {file_path}")
                try:
                    # Read the first line to determine the expected number of columns
                    with open(file_path, 'r') as f:
                        header_line = f.readline()
                        expected_columns = len(header_line.strip().split(';'))

                    # Check for malformed rows before loading the CSV
                    check_malformed_rows(file_path, expected_columns)

                    # Initialize a set to keep track of columns already checked in this file
                    columns_checked = set()

                    # Read the CSV file in chunks
                    chunksize = 300000  # Adjust this value based on your system's memory capacity
                    for chunk in pd.read_csv(file_path, delimiter=';', low_memory=False, chunksize=chunksize):
                        # Check each column in the chunk
                        for col in chunk.columns:
                            if col not in column_status:
                                column_status[col] = True  # Initially assume the column is empty/zero in all files

                            if col not in columns_checked:
                                columns_checked.add(col)  # Mark column as checked for this file

                            # If the column has already been marked as non-empty, skip checking
                            if not column_status[col]:
                                continue

                            # Check if the column in this chunk is entirely empty or zero-filled
                            if not is_empty_or_zero(chunk[col]):
                                column_status[col] = False  # If the column is not empty/zero, mark it as False
                except Exception as e:
                    print(f"Error while processing file {file_path}: {e}")

    # After processing all files, report the columns that are empty or contain only zeros/NaNs in all CSVs
    columns_to_report = [col for col, status in column_status.items() if status]

    if columns_to_report:
        print("\nThe following columns are empty or contain only zeros, NaNs, or False in all the analyzed CSVs:")
        for col in columns_to_report:
            print(f"- {col}")
    else:
        print("\nThere are no columns that are empty or contain only zeros, NaNs, or False in all the analyzed CSVs.")

if __name__ == "__main__":
    root_directory = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_4"
    analyze_csv_files(root_directory)