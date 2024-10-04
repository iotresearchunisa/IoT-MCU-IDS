from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], shuffle=True)

    # Preprocess the datasets
    X_train, y_train, scaler, label_encoder = preprocess_dataset(train_df, train=True)
    X_test, y_test, _, _ = preprocess_dataset(test_df, train=False, scaler=scaler, label_encoder=label_encoder)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Test Set Size: {len(test_df)}")

    return X_train, y_train, X_test, y_test, label_encoder


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, scaler=None, label_encoder=None):
    # 1. Separazione delle caratteristiche e del target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack']

    # 2. Mappatura della colonna 'Protocol_Type'
    protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'ARP': 4, 'MQTT': 5, 'SNA': 6, 'SSH': 7, 'SSHv2': 8, 'HPEXT': 9,
                    'DNS': 10, 'WiMax': 11, 'NTP': 12}

    X['Protocol_Type'] = X['Protocol_Type'].map(protocol_map)

    if X['Protocol_Type'].isnull().any():
        raise ValueError("Alcuni valori di 'Protocol_Type' non sono presenti nella mappa.")

    # 3. Label Encoding per la variabile target
    if train:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        if label_encoder is None:
            raise ValueError("LabelEncoder non fornito per i dati di test.")
        y_encoded = label_encoder.transform(y)

    if train:
        # Fit e transform sui dati di training
        scaler = MinMaxScaler()
        X_processed = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Preprocessor non fornito per i dati di test.")
        # Solo transform sui dati di test
        X_processed = scaler.transform(X)

    # Conversione a float32
    X_processed = X_processed.astype('float32')

    return X_processed, y_encoded, scaler, label_encoder