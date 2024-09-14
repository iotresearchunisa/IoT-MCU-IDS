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

def fill_or_replace_column(input_csv, output_csv, mode, column_name, string_to_insert, start_row, end_row):
    """
    This function fills or replaces values in a specified column of the CSV file, depending on the selected mode.

    :param input_csv: Path to the input CSV file.
    :param output_csv: Path to the output CSV file.
    :param mode: Operation mode - 'fill', 'replace', 'interval', or 'classify_zeros'.
    :param column_name: The name of the column to modify.
    :param string_to_insert: The string to insert in the column or to replace 0 values.
    :param start_row: Starting row for 'interval' mode (ignored for other modes).
    :param end_row: Ending row for 'interval' mode (ignored for other modes).
    """
    try:
        df = pd.read_csv(input_csv, delimiter=';', low_memory=False)

        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Error: The column '{column_name}' does not exist in the CSV file.")
            return

        if mode == 'interval':
            # Ensure the start and end rows are provided and valid
            if start_row is None or end_row is None:
                print("Error: You must specify a valid row interval.")
                return

            # Check if the provided row interval is valid
            if start_row < 1 or end_row > len(df) + 1:
                print("Error: The row interval is not valid.")
                return

            # Insert the string in the specified row interval (excluding the header)
            df.loc[start_row - 2:end_row - 2, column_name] = string_to_insert
            print(f"The column '{column_name}' was filled with the string '{string_to_insert}' in rows {start_row}-{end_row}.")

        elif mode == 'fill':
            # Fill the entire column with the specified string
            df[column_name] = string_to_insert
            print(f"The entire column '{column_name}' has been filled with the string '{string_to_insert}'.")

        elif mode == 'replace':
            # Replace all '0' or '0.0' values in the column with the specified string
            df[column_name] = df[column_name].astype(str).replace(['0', '0.0'], string_to_insert)
            print(f"Values '0' or '0.0' in the column '{column_name}' were replaced with '{string_to_insert}'.")

        elif mode == 'classify_zeros':
            # Function to classify sequences of zeros as 'attack' or 'benign'
            def classify_zeros(series):
                start_idx = -1
                zero_count = 0

                for i in range(len(series)):
                    if series[i] == 0 or series[i] == '0.0':
                        if start_idx == -1:
                            start_idx = i
                        zero_count += 1
                    else:
                        if zero_count > 0:
                            # Classify as 'attack' if the sequence of zeros is more than 6
                            if zero_count > 6:
                                series[start_idx:i] = ['attack'] * (i - start_idx)
                            else:
                                # Otherwise classify as 'benign'
                                series[start_idx:i] = ['benign'] * (i - start_idx)
                            zero_count = 0
                            start_idx = -1

                # Handle if the sequence of zeros is at the end of the series
                if zero_count > 0:
                    if zero_count > 6:
                        series[start_idx:] = ['attack'] * (len(series) - start_idx)
                    else:
                        series[start_idx:] = ['benign'] * (len(series) - start_idx)

                return series

            # Apply the classification function to the specified column
            df[column_name] = classify_zeros(df[column_name].tolist())

            # Find the indices of the rows with 0 or '0.0' values
            zero_indices = df[df[column_name].isin([0, '0', '0.0'])].index

            if not zero_indices.empty:
                for idx in zero_indices:
                    # Get 6 rows before and after the 0 values for context
                    start = max(0, idx - 6)  # Avoid negative index
                    end = min(len(df), idx + 7)  # Avoid going beyond the DataFrame length

                    print(f"Value '0' found in row {idx + 1}. Here are the nearby rows (column '{column_name}'):")
                    print(df.loc[start:end, column_name])
                    print("-" * 50)

        else:
            print("Error: The 'mode' parameter must be 'fill', 'replace', 'interval', or 'classify_zeros'.")
            return

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv, sep=';', index=False)
        print(f"File successfully saved to {output_csv}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    input_file = input("Enter the path to the input CSV file: ")
    output_file = input("Enter the path to the output CSV file: ")
    string_to_insert = ""

    print("Choose the type of operation:")
    print("1. Fill the entire column with a string.")
    print("2. Replace zeros in the column with a string.")
    print("3. Insert a string in a row interval (including the header).")
    print("4. Classify sequences of zeros as benign or attack.")

    choice = input("Enter the operation number (1, 2, 3, or 4): ")

    if choice != "4":
        # Ask for the string to insert for operations 1, 2, or 3
        string_to_insert = input("Enter 'benign' or 'attack': ")

        if string_to_insert not in ["benign", "attack"]:
            print("Error: Invalid string. Only 'benign' or 'attack' are allowed.")
            return

    if choice == '1':
        mode = 'fill'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert, start_row="", end_row="")
    elif choice == '2':
        mode = 'replace'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert, start_row="", end_row="")
    elif choice == '3':
        # Ask for the row interval
        start_row = int(input("Enter the starting row number (counting the header as row 1): "))
        end_row = int(input("Enter the ending row number (counting the header as row 1): "))
        mode = 'interval'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert, start_row=start_row, end_row=end_row)
    elif choice == '4':
        mode = 'classify_zeros'
        fill_or_replace_column(input_file, output_file, mode, "type_attack", string_to_insert, start_row="", end_row="")
    else:
        print("Error: Invalid choice.")

if __name__ == "__main__":
    main()

