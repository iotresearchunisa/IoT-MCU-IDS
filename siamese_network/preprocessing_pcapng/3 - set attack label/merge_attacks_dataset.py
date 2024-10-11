import os
import pandas as pd


def process_csv_files(root_dir, replacement_string, output_csv):
    csv_files = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))

    header_written = False

    for i, csv_file in enumerate(csv_files):
        print(f"Processing file {i + 1}/{len(csv_files)}: {csv_file}")

        chunksize = 300000

        try:
            reader = pd.read_csv(csv_file, sep=';', chunksize=chunksize, iterator=True, low_memory=False)
            for chunk in reader:
                # Replace 'attack', '0', or '0.0' with the replacement string in 'type_attack' column
                if 'type_attack' in chunk.columns:
                    chunk['type_attack'] = chunk['type_attack'].replace(['attack', '0', '0.0', 0, 0.0], replacement_string)

                if not header_written:
                    chunk.to_csv(output_csv, index=False, sep=';', mode='w')
                    header_written = True
                else:
                    chunk.to_csv(output_csv, index=False, sep=';', mode='a', header=False)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")


if __name__ == "__main__":
    root_dir = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_2/normal_attack/normal_scanning"
    replacement_string = "scanning"
    output_csv = "/mnt/FE9090E39090A3A5/Tesi/mio - TON_IoT/csv_cleaned_3/normal_attack/scanning.csv"

    process_csv_files(root_dir, replacement_string, output_csv)