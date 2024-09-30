import random
import numpy as np


# ==========================================================
#  Generate balanced Siamese pairs
# ==========================================================
def generate_balanced_siamese_pairs(data, labels, num_pairs):
    # Assicurarsi che num_pairs sia divisibile per 2
    assert num_pairs % 2 == 0, "Il numero di coppie deve essere divisibile per 2 per bilanciare le coppie positive e negative"

    # Separiamo i campioni per classe
    benign_data = data[labels == 0]
    attack_data = data[labels == 1]

    # Inizializzare le coppie e le etichette
    selected_pairs = []
    labels_for_pairs = []
    generated_pairs = set()  # Set per tracciare le coppie uniche

    # Numero di coppie positive e negative
    num_positive_pairs = num_pairs // 2
    num_negative_pairs = num_pairs // 2

    # Funzione per convertire una coppia in forma immutabile e ordinata
    def make_pair_key(pair):
        return tuple(sorted([tuple(pair[0]), tuple(pair[1])]))

    # Generare coppie positive
    while len(selected_pairs) < num_positive_pairs:
        # Generare una coppia positiva
        if random.random() < 0.5 and len(benign_data) >= 2:  # benign-benign
            pair = random.sample(list(benign_data), 2)
        elif len(attack_data) >= 2:  # attack-attack
            pair = random.sample(list(attack_data), 2)
        else:
            continue  # Se non ci sono abbastanza dati, salta

        # Creare una chiave ordinata per la coppia per evitare duplicati
        pair_key = make_pair_key(pair)

        # Se la coppia è già stata generata, saltare
        if pair_key in generated_pairs:
            continue

        # Aggiungere la coppia generata e la relativa etichetta
        selected_pairs.append(pair)
        labels_for_pairs.append(1)  # Coppia positiva

        # Aggiungere la coppia al set delle coppie generate
        generated_pairs.add(pair_key)

    # Generare coppie negative
    while len(selected_pairs) < num_positive_pairs + num_negative_pairs:
        # Generare una coppia negativa (benign-attack)
        pair = [random.choice(list(benign_data)), random.choice(list(attack_data))]

        # Creare una chiave ordinata per la coppia per evitare duplicati
        pair_key = make_pair_key(pair)

        # Se la coppia è già stata generata, saltare
        if pair_key in generated_pairs:
            continue

        # Aggiungere la coppia generata e la relativa etichetta
        selected_pairs.append(pair)
        labels_for_pairs.append(0)  # Coppia negativa

        # Aggiungere la coppia al set delle coppie generate
        generated_pairs.add(pair_key)

    return np.array(selected_pairs), np.array(labels_for_pairs)
