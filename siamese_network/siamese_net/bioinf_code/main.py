from siamese_net import siamese_network
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def preprocess_data(df):
    # Rimozione delle righe con 'Protocol_Type' non nella mappa
    protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'ARP': 4, 'MQTT': 5, 'SNA': 6, 'SSH': 7, 'SSHv2': 8,
                    'HPEXT': 9, 'DNS': 10, 'WiMax': 11, 'NTP': 12, "0xc0a8": 13, "HTTP": 14, "TLSv1": 15,
                    "WebSocket": 16, "TLSv1.2": 17, "RPCAP": 18, "HTTP/XML": 19, "SMTP": 20, "HTTP/JSON": 21,
                    "SSDP": 22, "FTP": 23, "TLSv1.1": 24, "MDNS": 25, "SSLv3": 26, "IMAP": 27, "IGMPv2": 28,
                    "POP": 29, "SMB": 30, "LLMNR": 31, "NBNS": 32, "RDP": 33, "IGMPv3": 34, "WHOIS": 35,
                    "BROWSER": 36, "FTP-DATA": 37, "TLSv1.3": 38, "SMB2": 39, "DCERPC": 40, "EPM": 41, "OCSP": 42,
                    "DHCPv6": 43, "BJNP": 44}

    valid_protocols = protocol_map.keys()
    valid_rows = df['Protocol_Type'].isin(valid_protocols)

    num_removed = (~valid_rows).sum()
    if num_removed > 0:
        print(f"Rimosse {num_removed} righe con valori non validi in 'Protocol_Type'.")

    df = df[valid_rows].reset_index(drop=True)

    type_attack_col = df[['type_attack']]

    # Mappatura della colonna 'Protocol_Type'
    df['Protocol_Type'] = df['Protocol_Type'].map(protocol_map)

    if df['Protocol_Type'].isnull().any():
        raise ValueError("Alcuni valori di 'Protocol_Type' non sono presenti nella mappa.")

    # Separa le colonne numeriche e la colonna Protocol_Type, escludendo type_attack
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler()
    df_preprocessed = scaler.fit_transform(df.drop(columns=['type_attack']))

    # Crea un DataFrame finale con le colonne trasformate (numeric scaled)
    column_names = list(numeric_cols)
    df_final = pd.DataFrame(df_preprocessed, columns=column_names)

    # Aggiungi la colonna type_attack non modificata al DataFrame finale
    df_final = pd.concat([type_attack_col.reset_index(drop=True), df_final], axis=1)

    return df_final


if __name__ == "__main__":
    csv_file = '../../datasets/mio/dataset_attacchi_con_MQTT_bilanciato.csv'
    path_result = "../results/bioinf_code/con_mqtt/"
    df = pd.read_csv(csv_file, sep=";")

    data = preprocess_data(df)

    y_df = data["type_attack"]
    del data["type_attack"]

    y = y_df.to_numpy()

    features_len = data.shape[1]
    input_shape = (features_len, 1)

    attack_type = pd.DataFrame(y, columns=['type_attack'])

    siamese_network(path_result, data, input_shape, features_len, attack_type)