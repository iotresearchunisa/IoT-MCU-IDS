import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer, Normalizer
from sklearn.neighbors import KNeighborsClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed()


# ==========================================================
#  Funzione di preprocessing
# ==========================================================
def preprocess_dataset(df, train=True, preprocessor=None):
    # 1. Separazione delle feature e del target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack'].apply(lambda x: 0 if x.lower() == 'benign' else 1).values

    # 3. Identificazione delle colonne per il preprocessing
    continuous_cols = ['rate', 'IAT', 'Time_To_Leave', 'Header_Length', 'Packet_Length', 'TCP_Length',
                       'MQTT_Length', 'MQTT_Proto_Length', 'MQTT_KeepAlive']

    count_cols = ['TCP_Flag_FIN', 'TCP_Flag_SYN', 'TCP_Flag_RST', 'TCP_Flag_PSH', 'TCP_Flag_ACK', 'TCP_Flag_ECE',
                  'TCP_Flag_CWR', 'Packet_Fragments', 'MQTT_HeaderFlags', 'MQTT_Reserved', 'MQTT_ConFlags',
                  'MQTT_CleanSession', 'MQTT_Retain', 'MQTT_WillFlag', 'MQTT_DupFlag', 'MQTT_Conflag_Retain']

    categorical_cols = ['Protocol_Type', 'MQTT_MessageType', 'MQTT_Conflag_QoS', 'MQTT_Version', 'MQTT_ConAck_Flags',
                        'MQTT_QoS']

    # Rimuovere eventuali colonne costanti dalle liste di colonne specifiche
    continuous_cols = [col for col in continuous_cols if col in df]
    count_cols = [col for col in count_cols if col in df]
    categorical_cols = [col for col in categorical_cols if col in df]


    # 4. Definizione della pipeline di preprocessing
    preprocessing_steps = []

    # Pipeline per feature numeriche continue con trasformazione logaritmica e RobustScaler
    if continuous_cols:
        continuous_pipeline = Pipeline([
            ('log_transform', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', RobustScaler())
        ])
        preprocessing_steps.append(('continuous', continuous_pipeline, continuous_cols))

    # Pipeline per feature di conteggio con MinMaxScaler
    if count_cols:
        count_pipeline = MinMaxScaler()
        preprocessing_steps.append(('count', count_pipeline, count_cols))

    # Pipeline per feature categoriali con OneHotEncoder
    if categorical_cols:
        categorical_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        preprocessing_steps.append(('categorical', categorical_pipeline, categorical_cols))

    # Creazione del ColumnTransformer con le trasformazioni specifiche
    column_transformer = ColumnTransformer(
        transformers=preprocessing_steps,
        remainder='drop'  # Drop any columns not specified
    )

    # Pipeline completa con ColumnTransformer e Normalizer
    preprocessor_pipeline = Pipeline([
        ('preprocessing', column_transformer),
        ('normalizer', Normalizer(norm='l2'))  # Aggiunta della normalizzazione
    ])

    if train:
        # Fit e transform sui dati di training
        X_processed = preprocessor_pipeline.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor non fornito per i dati di test.")
        # Solo transform sui dati di test
        X_processed = preprocessor.transform(X)

    # Conversione a float32
    X_processed = X_processed.astype('float32')

    return X_processed, y, preprocessor_pipeline


# ==========================================================
#  Carica e preprocessa il dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Identificazione delle colonne costanti
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"Rimozione delle colonne costanti: {constant_cols}\n")
        df = df.drop(columns=constant_cols)
    else:
        print("Nessuna colonna costante trovata.\n")

    # Divide in train/test prima del preprocessing
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], random_state=42)

    # Preprocessa i dataset
    X_train, y_train, preprocessor = preprocess_dataset(train_df, train=True)
    X_test, y_test, _ = preprocess_dataset(test_df, train=False, preprocessor=preprocessor)

    # Controlla la sovrapposizione tra set di train e test
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)

    print(f"Dimensione Set di Training: {len(train_df)}")
    print(f"Dimensione Set di Test: {len(test_df)}")

    assert train_indices.isdisjoint(test_indices), "I set di train e test si sovrappongono!"

    return X_train, y_train, X_test, y_test


# ==========================================================
#  Funzione principale per addestrare e valutare KNN
# ==========================================================
def train_knn(csv_file):
    # PRE-PROCESSING
    X_train, y_train, X_test, y_test = load_and_preprocess_data(csv_file, test_size=0.2)

    # TRAINING
    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Addestra il classificatore KNN
    knn_classifier.fit(X_train, y_train)

    # Valuta sul set di training
    y_train_pred = knn_classifier.predict(X_train)

    # Calcola le metriche di valutazione sul training
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)

    # Stampa i risultati sul training
    print("\nRisultati sul Training con KNN:")
    print(f"Accuracy del Training: {train_accuracy:.4f}")
    print(f"Precisione del Training: {train_precision:.4f}")
    print(f"Recall del Training: {train_recall:.4f}")
    print(f"F1 Score del Training: {train_f1:.4f}")

    # TESTING
    y_pred = knn_classifier.predict(X_test)

    # Calcola le metriche di valutazione sul test
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Stampa i risultati sul test
    print("\nRisultati sul Test con KNN:")
    print(f"Accuracy del Test: {test_accuracy:.4f}")
    print(f"Precisione del Test: {test_precision:.4f}")
    print(f"Recall del Test: {test_recall:.4f}")
    print(f"F1 Score del Test: {test_f1:.4f}")

    # Calcola e visualizza la matrice di confusione sul test
    cmatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMatrice di Confusione sul Test:")
    print(cmatrix)


if __name__ == "__main__":
    csv_file = '../../dataset_bilanciato.csv'
    train_knn(csv_file)
