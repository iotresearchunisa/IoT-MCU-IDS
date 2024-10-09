import os
import pandas as pd


def check_column_types_consistency(input_root):
    # Dictionary to store the column types from the first CSV as the reference
    reference_column_types = None
    inconsistent_files = []
    inconsistent_columns = {}

    # Traverse all directories and files in the input_root
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                # Use chunksize to read the CSV file in chunks
                chunks = pd.read_csv(input_file_path, delimiter=';', chunksize=300000, low_memory=False)

                # Process the first chunk to get the column types
                try:
                    first_chunk = next(chunks)
                    current_column_types = first_chunk.dtypes.to_dict()
                except StopIteration:
                    print(f"File {input_file_path} is empty.")
                    continue

                if reference_column_types is None:
                    # Store the first file's column types as the reference
                    reference_column_types = current_column_types
                    print(f"Reference column types set from: {input_file_path}")
                else:
                    # Compare the current file's column types with the reference
                    discrepancies_found = False  # Flag to track if any discrepancies are found

                    for column, dtype in current_column_types.items():
                        if column not in reference_column_types:
                            discrepancies_found = True
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' found in {input_file_path}, but not in reference")
                        elif reference_column_types[column] != dtype:
                            discrepancies_found = True
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' in {input_file_path} has type {dtype}, expected {reference_column_types[column]}")

                    # Find columns that are missing in the current file but exist in the reference
                    for column in reference_column_types:
                        if column not in current_column_types:
                            discrepancies_found = True
                            if column not in inconsistent_columns:
                                inconsistent_columns[column] = []
                            inconsistent_columns[column].append(f"Column '{column}' missing in {input_file_path}")

                    # Add the file to the inconsistent files list if there are any discrepancies
                    if discrepancies_found:
                        inconsistent_files.append(input_file_path)

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
    input_root = "/mnt/FE9090E39090A3A5/Tesi/TON_IoT/csv_cleaned_4"
    check_column_types_consistency(input_root)
