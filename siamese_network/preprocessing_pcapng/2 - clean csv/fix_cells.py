"""
This Python script processes and cleans large CSV files by applying transformations in chunks (blocks of data)
to optimize memory usage. The script handles comma-separated values and boolean values such as 'False'.
The main functionalities include:

1. **Processing CSV Files in Chunks**: The script reads and processes large CSV files in chunks of 100,000 rows
   at a time to reduce memory usage and handle large datasets efficiently.

2. **Extracting the Second Value**: For cells that contain multiple values separated by a comma (`,`),
   the script extracts and retains the second value. If the second value is 'False' or False, it is replaced with '0'.
   Empty cells or cells with a single value remain unchanged.

3. **Handling 'False' Values**: If a cell contains the value 'False' (either as a string or boolean),
   the script replaces it with '0'. This applies to cells with a single value or multiple values separated by commas.

4. **Counting Rows with Multiple Values**: The script identifies and counts rows that contain at least one cell
   with multiple comma-separated values, providing a summary of the rows processed.

5. **Incremental Saving of Transformed Data**: After transformation, the cleaned data is saved incrementally to
   an output CSV file, maintaining the original directory structure. The files are written progressively as each chunk
   is processed, optimizing performance.

### Usage:

- **Input**: Specify the directory path containing the CSV files to be processed. The script automatically scans
   all subdirectories to process each CSV file found.
- **Output**: The transformed CSV files are saved in the output directory, preserving the original folder structure.
- **Chunk Size**: To avoid memory issues, the script processes the CSV files in blocks of 100,000 rows at a time.

"""

import pandas as pd
import os


def extract_second_value(cell):
    if pd.isna(cell) or cell == '':
        return cell
    if ',' in cell:
        value = cell.split(',')[1]

        if value == "False" or value == False:
            return 0

        return value
    if cell == "False" or cell == False:
        return 0
    return cell


def process_csv_in_chunks(input_file_path, output_file_path, chunk_size=100000):
    total_rows_with_multiple_values = 0

    chunk_iterator = pd.read_csv(input_file_path, delimiter=';', low_memory=False, chunksize=chunk_size)

    with open(output_file_path, 'w', newline='') as output_file:
        for i, chunk in enumerate(chunk_iterator):
            rows_with_multiple_values = (chunk.astype(str).apply(lambda row: row.str.contains(',')).any(axis=1)).sum()
            total_rows_with_multiple_values += rows_with_multiple_values

            chunk_transformed = chunk.astype(str).apply(lambda col: col.apply(extract_second_value))

            chunk_transformed.to_csv(output_file, index=False, sep=';', mode='a', header=i == 0)

    print(f"The transformed CSV has been saved as {output_file_path}")
    print(f"Total number of rows with more than one comma-separated value: {total_rows_with_multiple_values}")


def process_all_csvs(input_root, output_root, chunk_size=100000):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                input_file_path = str(os.path.join(root, file))

                # Create the corresponding output file path
                relative_path = os.path.relpath(input_file_path, input_root)
                output_file_path = os.path.join(output_root, relative_path)

                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # Process the CSV file in chunks and save the output
                process_csv_in_chunks(input_file_path, output_file_path, chunk_size)


if __name__ == "__main__":
    input_root = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv/normal_attack/pp/"
    output_root = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned/normal_attack/"

    process_all_csvs(input_root, output_root, chunk_size=300000)
