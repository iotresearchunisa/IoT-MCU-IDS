import pandas as pd


def fill_or_replace_column(input_csv, output_csv, mode, column_name, string_to_insert, start_row, end_row):
    """
    Riempie una colonna con una stringa oppure sostituisce i valori 0 di una colonna con una stringa.

    :param input_csv: Path al file CSV di input.
    :param output_csv: Path al file CSV di output.
    :param column_name: Il nome della colonna da modificare.
    :param string_to_insert: La stringa da inserire nella colonna o al posto degli zeri.
    :param mode: 'fill' per riempire l'intera colonna, 'replace' per sostituire solo i valori 0, 'interval' per l'intervallo di righe.
    :param start_row: La riga iniziale (solo per 'interval').
    :param end_row: La riga finale (solo per 'interval').
    """

    try:
        # Carica il CSV nel dataframe
        df = pd.read_csv(input_csv, delimiter=';', low_memory=False)

        # Controlla se la colonna esiste
        if column_name not in df.columns:
            print(f"Errore: La colonna '{column_name}' non esiste nel file CSV.")
            return

        if mode == 'interval':
            # Controlla la validità dell'intervallo
            if start_row is None or end_row is None:
                print("Errore: È necessario specificare l'intervallo di righe.")
                return

            if start_row < 1 or end_row > len(df) + 1:
                print("Errore: L'intervallo di righe non è valido.")
                return

            # Applica l'intervallo alle righe dei dati, escludendo l'header
            df.loc[start_row - 2:end_row - 2, column_name] = string_to_insert  # Sottrai 1 a start_row per escludere l'header
            print(f"La colonna '{column_name}' è stata riempita con la stringa '{string_to_insert}' nell'intervallo {start_row}-{end_row}.")

        elif mode == 'fill':
            # Riempie l'intera colonna con la stringa specificata
            df[column_name] = string_to_insert
            print(f"Tutta la colonna '{column_name}' è stata riempita con la stringa '{string_to_insert}'.")

        elif mode == 'replace':
            # Converte la colonna in stringhe e sostituisce i valori '0' o '0.0' con la stringa specificata
            df[column_name] = df[column_name].astype(str).replace(['0', '0.0'], string_to_insert)
            print(f"I valori '0' o '0.0' nella colonna '{column_name}' sono stati sostituiti con '{string_to_insert}'.")

        elif mode == 'classify_zeros':

            # Funzione per classificare i valori 0
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
                            # Controlliamo se siamo all'interno dei limiti della lista
                            if zero_count > 6:
                                series[start_idx:i] = ['attack'] * (i - start_idx)
                            else:
                                series[start_idx:i] = ['benign'] * (i - start_idx)
                            zero_count = 0
                            start_idx = -1

                # Gestisce il caso in cui la serie termini con una sequenza di zeri
                if zero_count > 0:
                    if zero_count > 6:
                        series[start_idx:] = ['attack'] * (len(series) - start_idx)
                    else:
                        series[start_idx:] = ['benign'] * (len(series) - start_idx)

                return series

            # Applica la classificazione alla colonna specificata
            df[column_name] = classify_zeros(df[column_name].tolist())

            # Trova gli indici dei valori 0 o '0.0'
            zero_indices = df[df[column_name].isin([0, '0', '0.0'])].index

            if not zero_indices.empty:
                for idx in zero_indices:
                    # Ottieni 6 righe prima e 6 righe dopo la riga contenente 0
                    start = max(0, idx - 6)  # Evita di andare sotto l'indice 0
                    end = min(len(df), idx + 7)  # Evita di andare oltre la lunghezza del dataframe

                    print(f"Valore '0' trovato alla riga {idx + 1}. Ecco le righe vicine (colonna '{column_name}'):")
                    print(df.loc[start:end, column_name])
                    print("-" * 50)

        else:
            print("Errore: Il parametro 'mode' deve essere 'fill' o 'replace'.")
            return

        # Salva il CSV aggiornato
        df.to_csv(output_csv, sep=';', index=False)
        print(f"File salvato con successo in {output_csv}")

    except FileNotFoundError:
        print(f"Errore: Il file '{input_csv}' non esiste.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


def main():
    """
    Funzione principale per l'interazione con l'utente.
    """
    # Input da parte dell'utente
    input_file = input("Inserisci il percorso del file CSV di input: ")
    output_file = input("Inserisci il percorso del file CSV di output: ")
    stringa = ""

    # Modalità scelta dall'utente
    print("Scegli il tipo di operazione:")
    print("1. Riempire tutta la colonna con una stringa.")
    print("2. Sostituire i valori 0 nella colonna con una stringa.")
    print("3. Inserire una stringa in un intervallo di righe (contando l'header).")
    print("4. Classificare le sequenze di 0 come benign o attack.")

    scelta = input("Inserisci il numero dell'operazione (1, 2, 3 o 4): ")

    if scelta != "4":
        stringa = input("Inserisci 'benign' o 'attack': ")

        if stringa != "benign" and stringa != "attack":
            print("Stringa errata")
            return

    if scelta == '1':
        mod = 'fill'
        fill_or_replace_column(input_file, output_file, mod,"type_attack", stringa, start_row="", end_row="")
    elif scelta == '2':
        mod = 'replace'
        fill_or_replace_column(input_file, output_file, mod, "type_attack", stringa, start_row="", end_row="")
    elif scelta == '3':
        # Intervallo di righe
        start_row = int(input("Inserisci il numero della riga iniziale (contando l'header come riga 1): "))
        end_row = int(input("Inserisci il numero della riga finale (contando l'header come riga 1): "))
        mod = 'interval'
        fill_or_replace_column(input_file, output_file, mod,"type_attack", stringa, start_row=start_row, end_row=end_row)
    elif scelta == '4':
        mod = 'classify_zeros'
        fill_or_replace_column(input_file, output_file, mod, "type_attack", stringa, start_row="", end_row="")
    else:
        print("Errore: Scelta non valida.")


# Esegui il main solo se il file viene eseguito direttamente
if __name__ == "__main__":
    main()

