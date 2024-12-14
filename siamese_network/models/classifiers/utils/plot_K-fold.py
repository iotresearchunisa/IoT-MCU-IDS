import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_kfold(data, path, title):
    # Creazione del DataFrame
    df = pd.DataFrame(data)

    # Trattare 'Split' come variabile categoriale mantenendo l'ordine originale
    split_order = df['Split'].tolist()
    df['Split'] = pd.Categorical(df['Split'], categories=split_order, ordered=True)

    # Se la deviazione standard è in forma decimale e rappresenta una percentuale,
    # moltiplichiamola per 100 per allinearla con l'accuratezza media
    df['Std_knn'] = df['Std_knn'] * 100
    df['Std_rf'] = df['Std_rf'] * 100
    df['Std_svm'] = df['Std_svm'] * 100

    # Definizione dei classificatori e dei colori
    classificatori = {
        'knn': 'blue',
        'rf': 'green',
        'svm': 'red'
    }

    # Creazione di una figura con 3 sottotrame (subplot)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Titoli per ogni subplot
    titoli = {
        'knn': 'K-Nearest Neighbors (KNN)',
        'rf': 'Random Forest (RF)',
        'svm': 'Support Vector Machine (SVM)'
    }

    # Itera su ogni classificatore per creare i grafici
    for idx, (clf, color) in enumerate(classificatori.items()):
        ax = axes[idx]
        avg_col = f'Avg_{clf}'
        std_col = f'Std_{clf}'

        # Traccia la linea dell'accuratezza media
        sns.lineplot(x='Split', y=avg_col, data=df, label='Average Accuracy', color=color, marker='o', ax=ax)

        # Calcola i limiti superiore e inferiore per la deviazione standard
        upper = df[avg_col] + df[std_col]
        lower = df[avg_col] - df[std_col]

        # Aggiungi la fascia di deviazione standard
        ax.fill_between(df['Split'], lower, upper, color=color, alpha=0.2, label='Std Deviation')

        # Imposta il titolo del subplot
        ax.set_title(titoli[clf], fontsize=14)

        # Imposta le etichette degli assi
        ax.set_xlabel('Train data percentage', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)

        ax.legend()
        ax.grid(True)

    plt.suptitle(f'Avarage & Standard Deviation ({title})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    data_Few_ShoT = {
        'Split': ['64%', '48%', '4%', '0.07%'],
        'Avg_knn': [92.35, 92.34, 91.85, 87.75],
        'Std_knn': [0.0025, 0.0040, 0.0123, 0.0910],

        'Avg_rf': [92.26, 92.22, 90.10, 82.58],
        'Std_rf': [0.0016, 0.0022, 0.0131, 0.0802],

        'Avg_svm': [87.98, 87.93, 83.21, 76.98],
        'Std_svm': [0.0020, 0.0042, 0.0140, 0.1215]
    }

    data_TON_IoT = {
        'Split': ['64%', '48%', '4%', '0.07%'],
        'Avg_knn': [93.00, 92.95, 90.88, 77.27],
        'Std_knn': [0.0022, 0.0036, 0.0106, 0.1016],

        'Avg_rf': [93.48, 93.40, 91.33, 77.27],
        'Std_rf': [0.0020, 0.0021, 0.0108, 0.1095],

        'Avg_svm': [84.67, 84.51, 81.43, 75.45],
        'Std_svm': [0.0025, 0.0069, 0.0111, 0.1223]
    }

    plot_kfold(data_Few_ShoT, path="../../../results/plot/few-shot_iot.png", title="Few-Shot_IoT")
    plot_kfold(data_TON_IoT, path="../../../results/plot/ton_iot.png", title="TON_IoT")
