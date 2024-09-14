"""
This Python script processes and cleans CSV files by applying specific transformations to handle cells with multiple
comma-separated values, as well as counting the number of such rows. Its main functionalities include:

1. **Extracting Second Value**: If a cell contains multiple values separated by a comma (`,`), the script extracts and retains
   the second value. Empty cells are ignored.

2. **Counting Rows with Multiple Values**: The script identifies and counts rows that contain at least one cell with multiple
   comma-separated values.

3. **Saving Transformed Data**: After processing, the cleaned data is saved to a specified output CSV file.
"""

import pandas as pd
import argparse
import sys


def extract_second_value(cell):
    if pd.isna(cell) or cell == '':
        return cell  # Ignore empty values
    if ',' in cell:
        return cell.split(',')[1]
    return cell


def process_csv(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path, delimiter=';', low_memory=False)

    # Count rows that contain at least one cell with two values separated by a comma
    rows_with_multiple_values = (df.astype(str).apply(lambda row: row.str.contains(',')).any(axis=1)).sum()

    # Apply the function to extract the second value, ignoring empty values
    df_transformed = df.astype(str).apply(lambda col: col.apply(extract_second_value))

    # Save the transformed DataFrame to the output file
    df_transformed.to_csv(output_file_path, index=False, sep=';')

    print(f"The transformed CSV has been saved as {output_file_path}")
    print(f"Number of rows with more than one comma-separated value: {rows_with_multiple_values}")

    return rows_with_multiple_values


if __name__ == "__main__":
    input_file = "..."
    output_file = "..."

    process_csv(input_file, output_file)
