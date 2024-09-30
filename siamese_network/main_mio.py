from siamese_net_2 import siamese_network
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


def preprocess_data(df):
    type_attack_col = df[['type_attack']]

    # Separa le colonne numeriche e la colonna Protocol_Type, escludendo type_attack
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = ['Protocol_Type']

    # Definisci un ColumnTransformer per applicare MinMaxScaler alle colonne numeriche e OneHotEncoder alla colonna categoriale
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Applica il preprocessor al DataFrame (escludendo type_attack)
    df_preprocessed = preprocessor.fit_transform(df.drop(columns=['type_attack']))

    # Crea un DataFrame finale con le colonne trasformate (numeric scaled e one-hot encoded)
    column_names = list(numeric_cols) + list(
        preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
    df_final = pd.DataFrame(df_preprocessed, columns=column_names)

    # Aggiungi la colonna type_attack non modificata al DataFrame finale
    df_final = pd.concat([type_attack_col.reset_index(drop=True), df_final], axis=1)

    return df_final


if __name__ == "__main__":
    csv_file = 'dataset_attacchi_bilanciato.csv'
    df = pd.read_csv(csv_file, sep=";")

    data = preprocess_data(df)

    y_df = data["type_attack"]
    del data["type_attack"]

    y = y_df.to_numpy()

    features_len = data.shape[1]
    input_shape = (features_len, 1)

    attack_type = pd.DataFrame(y, columns=['type_attack'])

    siamese_network("codice_vecchio.txt", data, input_shape, features_len, attack_type)