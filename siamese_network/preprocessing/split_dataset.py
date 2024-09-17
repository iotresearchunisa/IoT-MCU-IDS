import pandas as pd
import logging

log_file_path = '../results/dataset_info.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


# Funzione per il conteggio dei campioni benign e attack
def count_samples(csv_file_path, chunk_size=200000, delimiter=';'):
    benign_count = 0
    attack_count = 0
    total_samples = 0

    logging.info('Inizio del conteggio dei campioni benign e attack.')

    for chunk_number, chunk in enumerate(pd.read_csv(csv_file_path, delimiter=delimiter, chunksize=chunk_size)):
        # Conta i campioni per chunk
        benign_chunk = (chunk['type_attack'] == 'benign').sum()
        attack_chunk = (chunk['type_attack'] == 'attack').sum()
        total_chunk = len(chunk)

        # Aggiorna i contatori totali
        benign_count += benign_chunk
        attack_count += attack_chunk
        total_samples += total_chunk

    # Calcola le percentuali
    benign_percentage = (benign_count / total_samples) * 100 if total_samples > 0 else 0
    attack_percentage = (attack_count / total_samples) * 100 if total_samples > 0 else 0

    # Log finale
    logging.info(f"Totale campioni benign: {benign_count}")
    logging.info(f"Totale campioni attack: {attack_count}")
    logging.info(f"Totale campioni: {total_samples}")
    logging.info(f"Percentuale benign: {benign_percentage:.2f}%")
    logging.info(f"Percentuale attack: {attack_percentage:.2f}%")

    # Stampa i risultati a schermo
    print(f"Numero di campioni benign: {benign_count}")
    print(f"Numero di campioni attack: {attack_count}")
    print(f"Totale campioni: {total_samples}")
    print(f"Percentuale benign: {benign_percentage:.2f}%")
    print(f"Percentuale attack: {attack_percentage:.2f}%")

    return benign_count, attack_count


# Funzione per creare un dataset bilanciato in modo ottimizzato e mescolato
def create_balanced_dataset(csv_file_path, output_file_path, chunk_size=100000, delimiter=';'):
    # Contiamo prima i campioni benign e attack
    benign_count, attack_count = count_samples(csv_file_path, chunk_size, delimiter)

    # Determiniamo il numero minimo di campioni tra benign e attack
    min_samples = min(benign_count, attack_count)
    logging.info(f"Numero minimo tra benign e attack per bilanciare: {min_samples}")

    benign_written = 0
    attack_written = 0
    header_written = False  # Variabile di controllo per l'header

    # Scriviamo i dati bilanciati per chunk
    for chunk_number, chunk in enumerate(pd.read_csv(csv_file_path, delimiter=delimiter, chunksize=chunk_size)):
        benign_chunk = chunk[chunk['type_attack'] == 'benign']
        attack_chunk = chunk[chunk['type_attack'] == 'attack']

        # Campiona casualmente solo i campioni necessari fino a raggiungere il min_samples
        if benign_written < min_samples and not benign_chunk.empty:
            benign_to_write = benign_chunk.sample(n=min(min_samples - benign_written, len(benign_chunk)), random_state=42)
            benign_written += len(benign_to_write)
        else:
            benign_to_write = pd.DataFrame()

        if attack_written < min_samples and not attack_chunk.empty:
            attack_to_write = attack_chunk.sample(n=min(min_samples - attack_written, len(attack_chunk)), random_state=42)
            attack_written += len(attack_to_write)
        else:
            attack_to_write = pd.DataFrame()

        # Mescola i campioni benign e attack selezionati
        combined_chunk = pd.concat([benign_to_write, attack_to_write]).sample(frac=1, random_state=42)

        # Scrivi il chunk mescolato nel file CSV
        combined_chunk.to_csv(output_file_path, mode='a', header=not header_written, index=False, sep=delimiter)

        # Imposta la variabile header_written su True dopo il primo chunk
        header_written = True

        # Ferma la scrittura una volta raggiunto il numero minimo di campioni per entrambi i tipi
        if benign_written >= min_samples and attack_written >= min_samples:
            break

    logging.info(f"Dataset bilanciato creato con {min_samples} campioni benign e {min_samples} campioni attack.")
    logging.info(f"Dataset bilanciato salvato in: {output_file_path}")

    print(f"Dataset bilanciato creato con {min_samples} campioni benign e {min_samples} campioni attack.")
    print(f"Dataset bilanciato salvato in: {output_file_path}")


def main():
    csv_file_path = '../../dataset/dataset.csv'
    balanced_csv_file_path = '../../dataset/dataset_bilanciato.csv'

    print("Scegli un'opzione:")
    print("1. Contare i campioni benign e attack")
    print("2. Bilanciare il dataset")

    scelta = input("Inserisci il numero della tua scelta: ")

    if scelta == "1":
        count_samples(csv_file_path)
    elif scelta == "2":
        create_balanced_dataset(csv_file_path, balanced_csv_file_path)
    else:
        print("Scelta non valida. Riprova.")


if __name__ == "__main__":
    main()