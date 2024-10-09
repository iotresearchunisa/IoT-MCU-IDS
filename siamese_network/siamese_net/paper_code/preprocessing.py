import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], shuffle=True)

    # Split train_val_df into train/val
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['type_attack'], shuffle=True)

    # Check for overlap between train, val, and test sets
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)

    assert train_indices.isdisjoint(val_indices), "Train and validation sets overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and test sets overlap!"
    assert val_indices.isdisjoint(test_indices), "Validation and test sets overlap!"

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Preprocess the datasets
    X_train, y_train, scaler, label_encoder = preprocess_dataset(train_df, train=True)
    X_val, y_val, _, _ = preprocess_dataset(val_df, train=False, scaler=scaler, label_encoder=label_encoder)
    X_test, y_test, _, _ = preprocess_dataset(test_df, train=False, scaler=scaler, label_encoder=label_encoder)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, label_encoder


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, scaler=None, label_encoder=None):
    # 1. Rimozione delle righe con 'Protocol_Type' non nella mappa
    # mio dataset
    '''protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'ARP': 4, 'MQTT': 5,
                    'SNA': 6, 'SSH': 7, 'SSHv2': 8, 'HPEXT': 9,
                    'DNS': 10, 'WiMax': 11, 'NTP': 12}'''

    # TON_IoT dataset
    protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'ARP': 4, 'MQTT': 5, 'SSH': 7, 'SSHv2': 8,
                    'DNS': 10, 'NTP': 12, "0xc0a8": 13, "HTTP": 14, "TLSv1": 15, "WebSocket": 16,
                    "TLSv1.2": 17, "RPCAP": 18, "HTTP/XML": 19, "SMTP": 20, "HTTP/JSON": 21, "SSDP": 22,
                    "FTP": 23, "TLSv1.1": 24, "MDNS": 25, "SSLv3": 26, "IMAP": 27, "IGMPv2": 28, "POP": 29,
                    "SMB": 30, "LLMNR": 31, "NBNS": 32, "RDP": 33, "IGMPv3": 34, "WHOIS": 35, "BROWSER": 36,
                    "FTP-DATA": 37, "TLSv1.3": 38, "SMB2": 39, "DCERPC": 40, "EPM": 41, "OCSP": 42,
                    "DHCPv6": 43, "BJNP": 44}

    valid_protocols = protocol_map.keys()
    valid_rows = df['Protocol_Type'].isin(valid_protocols)

    num_removed = (~valid_rows).sum()
    if num_removed > 0:
        print(f"Rimosse {num_removed} righe con valori non validi in 'Protocol_Type'.")

    df = df[valid_rows].reset_index(drop=True)

    # 2. Separazione delle caratteristiche e del target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack']

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