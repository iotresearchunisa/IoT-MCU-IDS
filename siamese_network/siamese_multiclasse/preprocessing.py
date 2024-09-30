import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], random_state=42)

    # Split train_val_df into train/val
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['type_attack'], random_state=42)

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
    X_train, y_train, preprocessor_pipeline, label_encoder = preprocess_dataset(train_df, train=True)
    X_val, y_val, _, _ = preprocess_dataset(val_df, train=False, preprocessor_pipeline=preprocessor_pipeline, label_encoder=label_encoder)
    X_test, y_test, _, _ = preprocess_dataset(test_df, train=False, preprocessor_pipeline=preprocessor_pipeline, label_encoder=label_encoder)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_encoder


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, preprocessor_pipeline=None, label_encoder=None):
    # 1. Separazione delle caratteristiche e del target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack']

    # 2. Label Encoding per la variabile target
    if train:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        if label_encoder is None:
            raise ValueError("LabelEncoder non fornito per i dati di test.")
        y_encoded = label_encoder.transform(y)

    # 3. Identificazione delle colonne testuali e numeriche
    text_col = 'Protocol_Type'
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # OneHotEncoder per la colonna testuale (Protocol_Type)
    text_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # MinMaxScaler per le colonne numeriche
    numeric_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # ColumnTransformer per combinare le pipeline testuali e numeriche
    column_transformer = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, [text_col]),
            ('numeric', numeric_pipeline, numeric_cols)
        ],
        remainder='drop'
    )

    # Pipeline completa con ColumnTransformer
    preprocessor = Pipeline([
        ('preprocessing_pcapng', column_transformer)
    ])

    if train:
        # Fit e transform sui dati di training
        X_processed = preprocessor.fit_transform(X)
    else:
        if preprocessor_pipeline is None:
            raise ValueError("Preprocessor non fornito per i dati di test.")
        # Solo transform sui dati di test
        X_processed = preprocessor_pipeline.transform(X)

    # Conversione a float32
    X_processed = X_processed.astype('float32')

    return X_processed, y, preprocessor, label_encoder