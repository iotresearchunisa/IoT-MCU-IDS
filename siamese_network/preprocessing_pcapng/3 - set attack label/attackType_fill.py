"""
This Python script performs several operations on a specified column of a CSV file. It supports the following modes:

1. **Fill the Column**: Fills the entire specified column with a provided string.
2. **Replace Zeros**: Replaces all occurrences of `0` or `0.0` in the specified column with a given string.
3. **Insert in a Row Interval**: Inserts a given string in a specified row range for the specified column.
4. **Classify Zeros**: Classifies sequences of zeros in the specified column as either 'benign' or 'attack'.
   - Sequences longer than 6 are marked as 'attack'.
   - Sequences of 6 or fewer zeros are marked as 'benign'.
"""

import pandas as pd
import os

def fill_or_replace_column(input_csv, output_csv, mode, column_name, string_to_insert=None, start_row=None, end_row=None):
    try:
        chunksize = 300000  # Adjust this based on your system's memory capacity
        header_written = False  # To write the header only once

        # Remove output file if it exists to prevent appending to old data
        if os.path.exists(output_csv):
            os.remove(output_csv)

        # For 'classify_zeros' mode, we need to keep track of sequences across chunks
        if mode == 'classify_zeros':
            # Read the specific column into memory
            try:
                column_data = pd.read_csv(input_csv, delimiter=';', usecols=[column_name], low_memory=False)[column_name]
            except ValueError:
                print(f"Error: The column '{column_name}' does not exist in the CSV file.")
                return

            # Convert the column to string type to handle mixed data types
            column_data = column_data.astype(str).reset_index(drop=True)

            # Initialize variables
            zero_count = 0
            indices_to_update = []

            for idx, value in enumerate(column_data):
                if value in ['0', '0.0', 0, 0.0]:
                    zero_count += 1
                    indices_to_update.append(idx)
                else:
                    if zero_count > 0:
                        classification = 'attack' if zero_count > 6 else 'benign'
                        column_data.iloc[indices_to_update] = classification
                        zero_count = 0
                        indices_to_update = []
                    # No change to value
            # After processing all values, handle any remaining zeros
            if zero_count > 0:
                classification = 'attack' if zero_count > 6 else 'benign'
                column_data.iloc[indices_to_update] = classification

            # Now, write back the modified column to the CSV
            # Read the entire CSV in chunks and replace the column
            for chunk in pd.read_csv(input_csv, delimiter=';', low_memory=False, chunksize=chunksize):
                # Replace the 'type_attack' column in the chunk
                start_idx = chunk.index[0]
                end_idx = chunk.index[-1]
                chunk[column_name] = column_data.iloc[start_idx:end_idx+1].values

                # Write the processed chunk to the output CSV file
                if not header_written:
                    chunk.to_csv(output_csv, sep=';', index=False, mode='w')
                    header_written = True
                else:
                    chunk.to_csv(output_csv, sep=';', index=False, mode='a', header=False)

            print(f"File successfully saved to {output_csv}")

        else:
            total_rows_processed = 0  # Keep track of the total number of rows processed

            # Read and process the CSV in chunks
            for chunk in pd.read_csv(input_csv, delimiter=';', low_memory=False, chunksize=chunksize):
                # Check if the specified column exists
                if column_name not in chunk.columns:
                    print(f"Error: The column '{column_name}' does not exist in the CSV file.")
                    return

                if mode == 'fill':
                    # Fill the entire column with the specified string
                    chunk[column_name] = string_to_insert

                elif mode == 'replace':
                    # Replace all '0' or '0.0' values in the column with the specified string
                    chunk[column_name] = chunk[column_name].astype(str).replace(['0', '0.0'], string_to_insert)

                elif mode == 'interval':
                    # Calculate the chunk's start and end indices relative to the entire dataset
                    chunk_start = total_rows_processed + 1  # +1 because CSV rows start from 1
                    chunk_end = total_rows_processed + len(chunk)

                    # Check if there's an overlap with the specified interval
                    if start_row <= chunk_end and end_row >= chunk_start:
                        # Calculate the overlapping indices
                        overlap_start = max(0, start_row - chunk_start)
                        overlap_end = min(len(chunk), end_row - chunk_start + 1)

                        # Insert the string in the overlapping interval
                        chunk.iloc[overlap_start:overlap_end, chunk.columns.get_loc(column_name)] = string_to_insert

                else:
                    print("Error: The 'mode' parameter must be 'fill', 'replace', 'interval', or 'classify_zeros'.")
                    return

                # Write the processed chunk to the output CSV file
                if not header_written:
                    chunk.to_csv(output_csv, sep=';', index=False, mode='w')
                    header_written = True
                else:
                    chunk.to_csv(output_csv, sep=';', index=False, mode='a', header=False)

                total_rows_processed += len(chunk)

            print(f"File successfully saved to {output_csv}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    input_file = input("Enter the path to the input CSV file: ")
    output_file = input("Enter the path to the output CSV file: ")

    print("Choose the type of operation:")
    print("1. Fill the entire column with a string.")
    print("2. Replace zeros in the column with a string.")
    print("3. Insert a string in a row interval (including the header).")
    print("4. Classify sequences of zeros as benign or attack.")

    choice = input("Enter the operation number (1, 2, 3, or 4): ")
    string_to_insert = ""

    if choice in ["1", "2", "3"]:
        string_to_insert = input("Enter 'benign' or 'attack': ")

        if string_to_insert not in ["benign", "attack"]:
            print("Error: Invalid string. Only 'benign' or 'attack' are allowed.")
            return

    if choice == '1':
        mode = 'fill'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert)
    elif choice == '2':
        mode = 'replace'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert)
    elif choice == '3':
        start_row = int(input("Enter the starting row number (counting the header as row 1): "))
        end_row = int(input("Enter the ending row number (counting the header as row 1): "))
        mode = 'interval'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert, start_row=start_row, end_row=end_row)
    elif choice == '4':
        mode = 'classify_zeros'
        fill_or_replace_column(input_file, output_file, mode, "type_attack")
    else:
        print("Error: Invalid choice.")

if __name__ == "__main__":
    main()
