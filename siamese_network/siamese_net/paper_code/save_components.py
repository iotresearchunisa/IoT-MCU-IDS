import pickle


# ==========================================================
#  Save and load components
# ==========================================================
def load_components(preprocessing_path):
    with open(preprocessing_path + "preprocessing.pkl", 'rb') as prepro_file:
        scaler, label_encoder = pickle.load(prepro_file)
    print(f"Preprocessing caricato da {preprocessing_path}preprocessing.pkl")

    return scaler, label_encoder


def save_components(model, scaler, le, results_path):
    model.save(results_path + 'siamese_model.h5')
    model.save(results_path + 'siamese_model.keras')
    print(f"Modello salvato in {results_path}")

    with open(results_path + "preprocessing.pkl", 'wb') as file:
        pickle.dump((scaler, le), file)
    print("Pre-processing salvato!")


def save_pairs_to_header(pairs_a, pairs_b, labels, filename, subset='SIAMESE'):
    num_pairs = pairs_a.shape[0]
    feature_size = pairs_b.shape[1]  # 31

    with open(filename, 'w') as f:
        # Header guard
        header_guard = f"{subset.upper()}_MODEL_PAIRS_H"
        f.write(f'#ifndef {header_guard}\n')
        f.write(f'#define {header_guard}\n\n')

        # Definisci FEATURE_SIZE e NUM_{SUBSET}_MODEL_PAIRS come macro
        f.write(f'#define FEATURE_SIZE {feature_size}\n')
        f.write(f'#define NUM_PAIRS {num_pairs}\n\n')

        # Salva le coppie di dati in un unico array (pairs_a)
        f.write(f'const float pairs_a[NUM_PAIRS][FEATURE_SIZE][1][1] = {{\n')
        for i in range(num_pairs):
            f.write('  {\n')  # Inizio del campione
            for j in range(feature_size):
                value = pairs_a[i][j][0][0]
                # Formatta senza la virgola se è l'ultimo elemento
                f.write(f'    {{ {{{value:.8e}}} }}' + (', ' if j < feature_size - 1 else ''))
            f.write('\n  }' + (',\n' if i < num_pairs - 1 else '\n'))
        f.write('};\n\n')

        # Salva le coppie di dati in un unico array (pairs_b)
        f.write(f'const float pairs_b[NUM_PAIRS][FEATURE_SIZE][1][1] = {{\n')
        for i in range(num_pairs):
            f.write('  {\n')  # Inizio del campione
            for j in range(feature_size):
                value = pairs_b[i][j][0][0]
                # Formatta senza la virgola se è l'ultimo elemento
                f.write(f'    {{ {{{value:.8e}}} }}' + (', ' if j < feature_size - 1 else ''))
            f.write('\n  }' + (',\n' if i < num_pairs - 1 else '\n'))
        f.write('};\n\n')

        # Salva le etichette come array unidimensionale
        labels_str = ', '.join(map(str, labels))
        f.write(f'const int {subset.upper()}_MODEL_LABELS[NUM_PAIRS] = {{{labels_str}}};\n\n')

        # Fine dell'header guard
        f.write(f'#endif // {header_guard}\n')


