import os
import pandas as pd

def process_csv_files(root_dir, replacement_string, output_csv):
    csv_files = []

    # Collect all CSV files from the root directory and its subdirectories
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))

    # Initialize a variable to check if header is written
    header_written = False

    # Process each CSV file
    for i, csv_file in enumerate(csv_files):
        print(f"Processing file {i + 1}/{len(csv_files)}: {csv_file}")
        # Read the CSV in chunks to optimize memory usage
        chunksize = 100000  # Adjust the chunk size as needed
        try:
            reader = pd.read_csv(csv_file, sep=';', chunksize=chunksize, iterator=True)
            for chunk in reader:
                # Replace 'attack' with the replacement string in 'type_attack' column
                if 'type_attack' in chunk.columns:
                    chunk['type_attack'] = chunk['type_attack'].replace('attack', replacement_string)

                # Write to the output CSV file
                if not header_written:
                    chunk.to_csv(output_csv, index=False, sep=';', mode='w')
                    header_written = True
                else:
                    chunk.to_csv(output_csv, index=False, sep=';', mode='a', header=False)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

if __name__ == "__main__":
    root_dir = "/home/alberto/Documenti/GitHub/Thesis-IoT_Cloud_based/dataset/csv_cleaned_4/attacks/recon"
    replacement_string = "recon"
    output_csv = "/media/alberto/DATA/Tesi/recon/recon.csv"

    process_csv_files(root_dir, replacement_string, output_csv)
