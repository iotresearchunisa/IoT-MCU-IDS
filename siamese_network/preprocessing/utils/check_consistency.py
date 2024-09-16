"""
This Python script checks for column type consistency across all CSV files in a specified directory and its subdirectories.
It identifies columns with inconsistent data types when compared to the first CSV file processed (used as a reference).
The script generates a report at the end, showing which columns and files are inconsistent.

### Main functionalities:
1. **Column Type Consistency Check**: The script compares the data types of columns in each CSV file against the
   reference CSV file. It reports any column with inconsistent data types or missing columns.

2. **Handling New or Missing Columns**: If a file contains columns that are not present in the reference, or if a file
   is missing columns that are present in the reference, the script logs these discrepancies.

3. **Recursive Processing**: The script processes all CSV files in the input directory and its subdirectories.

### Usage:
- **Input**: Specify the path of the input directory containing the CSV files.
- **Output**: The script prints a detailed report of any column type inconsistencies across the files.
"""

import os
import pandas as pd

def check_column_types_consistency(input_root):
    # Dictionary to store the column types from the first CSV as the reference
    reference_column_types = None
    inconsistent_files = []
    inconsistent_columns = {}

    # Traverse all directories and files in the input_root
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                df = pd.read_csv(input_file_path, delimiter=';', low_memory=False)

                # Get the data types of the columns
                current_column_types = df.dtypes.to_dict()

                if reference_column_types is None:
                    # Store the first file's column types as the reference
                    reference_column_types = current_column_types
                    print(f"Reference column types set from: {input_file_path}")
                else:
                    # Compare the current file's column types with the reference
                    for column, dtype in current_column_types.items():
                        if column not in reference_column_types:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' found in {input_file_path}, but not in reference")
                        elif reference_column_types[column] != dtype:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' in {input_file_path} has type {dtype}, expected {reference_column_types[column]}")

                    # Find columns that are missing in the current file but exist in the reference
                    for column in reference_column_types:
                        if column not in current_column_types:
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' missing in {input_file_path}")

                    # Add the file to the inconsistent files list if there are any discrepancies
                    if any(column in inconsistent_columns for column in current_column_types):
                        inconsistent_files.append(input_file_path)

    # Final report
    if inconsistent_columns:
        print("\nInconsistent columns found:")
        for column, messages in inconsistent_columns.items():
            print(f"\nColumn: {column}")
            for message in messages:
                print(f"  - {message}")
    else:
        print("\nAll files have consistent column types.")

    if inconsistent_files:
        print("\nFiles with inconsistencies:")
        for file in inconsistent_files:
            print(f"- {file}")
    else:
        print("\nAll files are consistent.")

if __name__ == "__main__":
    input_root = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_4"

    # Check column type consistency across all CSV files
    check_column_types_consistency(input_root)
