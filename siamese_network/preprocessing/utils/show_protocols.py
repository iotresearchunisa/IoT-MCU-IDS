import csv


def get_unique_protocol_types(input_csv):
    """
    Function to return the unique values from the 'Protocol_Type' column in a CSV file.

    :param input_csv: Path to the input CSV file.
    :return: Set of unique values from the 'Protocol_Type' column.
    """
    protocols = set()

    try:
        # Open the input CSV file for reading with delimiter `;`
        with open(input_csv, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=';')

            # Read the header and find the index of the 'Protocol_Type' column
            header = next(reader)
            protocol_index = header.index('Protocol_Type')

            # Iterate through each row and collect unique 'Protocol_Type' values
            for row in reader:
                protocols.add(row[protocol_index])

        return protocols

    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    input_fie = "input.csv"

    unique_protocols = get_unique_protocol_types(input_fie)

    if unique_protocols:
        print(f"Unique Protocol Types in '{input_fie}':")
        for protocol in unique_protocols:
            print(protocol)
