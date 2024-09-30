from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing_pcapng
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], random_state=42)

    # Preprocess the datasets
    X_train, y_train, preprocessor = preprocess_dataset(train_df, train=True)
    X_test, y_test, _ = preprocess_dataset(test_df, train=False, preprocessor=preprocessor)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Test Set Size: {len(test_df)}")

    return X_train, y_train, X_test, y_test


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, preprocessor=None):
    # 1. Separation of features and target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack'].apply(lambda x: 0 if x.lower() == 'benign' else 1).values

    # 2. Identification of text and numeric columns
    text_col = 'Protocol_Type'
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # OneHotEncoder per la colonna di testo (Protocol_Type)
    text_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # MinMaxScaler per le colonne numeriche
    numeric_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # ColumnTransformer per combinare le pipeline di testo e numeriche
    column_transformer = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, [text_col]),  # One-hot encoding per la colonna Protocol_Type
            ('numeric', numeric_pipeline, numeric_cols)  # Min-Max Scaling per le colonne numeriche
        ],
        remainder='drop'  # Droppa eventuali colonne non specificate
    )

    # Pipeline completa con ColumnTransformer
    preprocessor_pipeline = Pipeline([
        ('preprocessing_pcapng', column_transformer)
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

