from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], shuffle=True)

    # Preprocess the datasets
    X_train, y_train, preprocessor, label_encoder = preprocess_dataset(train_df, train=True)
    X_test, y_test, _, _ = preprocess_dataset(test_df, train=False, preprocessor=preprocessor, label_encoder=label_encoder)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Test Set Size: {len(test_df)}")

    return X_train, y_train, X_test, y_test, label_encoder


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, preprocessor=None, label_encoder=None):
    # 1. Separation of features and target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack']

    # 2. Label Encoding for multi-class target variable
    if train:
        # Fit LabelEncoder on training labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        if label_encoder is None:
            raise ValueError("LabelEncoder not provided for test data.")
        # Transform test labels using the fitted LabelEncoder
        y_encoded = label_encoder.transform(y)

    # 3. Identification of text and numeric columns
    text_col = 'Protocol_Type'
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # OneHotEncoder for the text column (Protocol_Type)
    text_pipeline = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # MinMaxScaler for numeric columns
    numeric_pipeline = Pipeline([
        ('scaler', MinMaxScaler())
    ])

    # ColumnTransformer to combine text and numeric pipelines
    column_transformer = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, [text_col]),  # One-hot encoding for 'Protocol_Type'
            ('numeric', numeric_pipeline, numeric_cols)  # Min-Max Scaling for numeric columns
        ],
        remainder='drop'  # Drop any unspecified columns
    )

    # Complete pipeline with ColumnTransformer
    preprocessor_pipeline = Pipeline([
        ('preprocessing', column_transformer)
    ])

    if train:
        # Fit and transform on training data
        X_processed = preprocessor_pipeline.fit_transform(X)
    else:
        if preprocessor is None:
            raise ValueError("Preprocessor not provided for test data.")
        # Only transform on test data
        X_processed = preprocessor.transform(X)

    # Convert to float32 for efficiency
    X_processed = X_processed.astype('float32')

    return X_processed, y_encoded, preprocessor_pipeline, label_encoder
