from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from generate_pairs import *
from preprocessing import *
from siamese_net import SiameseNet
import pandas as pd
from save_components import *


# ==========================================================
#  Check for GPU or CPU
# ==========================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Trovate {len(gpus)} GPU:")
    for gpu in gpus:
        print(f" - {gpu.name}")
else:
    print("Nessuna GPU trovata. TensorFlow utilizzerà la CPU.")


# ==========================================================
#  Main training function
# ==========================================================
def train_siamese_network(csv_file, path_results, model_pretrained, num_pairs, save_pairs, convert_model):
    ################################################################################################################
    print("\n================= [STEP 1.0] Load and preprocess the datasets =================")
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['type_attack'], shuffle=True)

    # Split train_val_df into train/val
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['type_attack'],
                                        shuffle=True)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    scaler, label_encoder = load_components(model_pretrained)

    # Preprocess the datasets
    X_train, y_train, _, _ = preprocess_dataset(val_df, train=False, scaler=scaler, label_encoder=label_encoder)
    X_val, y_val, _, _ = preprocess_dataset(val_df, train=False, scaler=scaler, label_encoder=label_encoder)
    X_test, y_test, _, _ = preprocess_dataset(test_df, train=False, scaler=scaler, label_encoder=label_encoder)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    print("Data are preprocessed!")
    ################################################################################################################


    ################################################################################################################
    # Generation of pairs
    print("\n================= [STEP 2.0] Generation of pairs =================")
    train_pairs, train_labels = generate_balanced_siamese_pairs(X_train, y_train, num_pairs=num_pairs[0])
    val_pairs, val_labels = generate_balanced_siamese_pairs(X_val, y_val, num_pairs=num_pairs[1])
    test_pairs, test_labels = generate_balanced_siamese_pairs(X_test, y_test, num_pairs=num_pairs[2])
    print("Pairs are generated!")

    # Check pairs duplicated
    print("\n================= [STEP 2.1] Check pairs duplicated =================")
    train_identic_count = np.sum(np.all(train_pairs[:, 0] == train_pairs[:, 1], axis=tuple(range(1, train_pairs[:, 0].ndim))))
    val_identic_count = np.sum(np.all(val_pairs[:, 0] == val_pairs[:, 1], axis=tuple(range(1, val_pairs[:, 0].ndim))))
    test_identic_count = np.sum(np.all(test_pairs[:, 0] == test_pairs[:, 1], axis=tuple(range(1, test_pairs[:, 0].ndim))))

    print(f"Numero di coppie identiche TRAIN SET: {train_identic_count}/{len(train_pairs)}")
    print(f"Numero di coppie identiche VAL SET: {val_identic_count}/{len(val_pairs)}")
    print(f"Numero di coppie identiche TEST SET: {test_identic_count}/{len(test_pairs)}")

    # Reshape pairs
    print("\n================= [STEP 2.2] Reshape pairs in (x, 31, 1,1) =================")
    train_a = train_pairs[:, 0].reshape(-1, 31, 1, 1)
    train_b = train_pairs[:, 1].reshape(-1, 31, 1, 1)
    val_a = val_pairs[:, 0].reshape(-1, 31, 1, 1)
    val_b = val_pairs[:, 1].reshape(-1, 31, 1, 1)
    test_a = test_pairs[:, 0].reshape(-1, 31, 1, 1)
    test_b = test_pairs[:, 1].reshape(-1, 31, 1, 1)
    print("Pairs are reshaped!")

    # Save pairs in test_pairs.h
    if save_pairs:
        print("\n================= [STEP 2.3] Save pairs in 'test_pairs.h' =================")
        save_pairs_to_header(test_a, test_b, test_labels, path_results + 'test_pairs.h')
        print(f"Pairs are saved in {path_results}test_pairs.h")
    ################################################################################################################


    ################################################################################################################
    print("\n================= [STEP 3.0] Training Phase =================")
    # Create Siamese model
    siamese_model = (SiameseNet(input_shape=(X_train.shape[1], 1, 1))).load_saved_model(model_pretrained + "siamese_model.h5")

    # Define early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = siamese_model.fit([train_a, train_b], train_labels,
                                validation_data=([val_a, val_b], val_labels),
                                batch_size=256,
                                epochs=40,
                                callbacks=[early_stopping])

    # Save components
    save_components(siamese_model, scaler, label_encoder, path_results)

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_results + "training_loss.png")
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path_results + "training_acc.png")
    plt.show()
    ################################################################################################################


    ################################################################################################################
    # Evaluate the model on unseen test data
    print("\n================= [STEP 4.0] Evaluate Model =================")
    test_loss, test_accuracy = siamese_model.evaluate([test_a, test_b], test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predictions on test set
    predictions = siamese_model.predict([test_a, test_b]).ravel()
    predicted_labels = (predictions < 0.5).astype(int)

    # Evaluate the precision, recall, F1-score
    print(classification_report(test_labels, predicted_labels))
    ################################################################################################################


    ################################################################################################################
    if convert_model:
        print("\n================= [STEP 5.0] Model conversion =================")
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(siamese_model)
        tflite_model = converter.convert()

        # Save the model.
        with open(path_results + 'model.tflite', 'wb') as f:
            f.write(tflite_model)
        print("Model converted!")
    ################################################################################################################


if __name__ == "__main__":
    csv_file = '../../datasets/TON_IoT/dataset.csv'
    result_path = "../results/paper_code/test/mio - TON_IoT/transfer_learning/con_mqtt/"
    model_pretrained = "../results/paper_code/train_test/con_mqtt/"

    convert_model = False
    save_pairs = False

    # Train - Val - Test
    num_pairs = [1000000, 1000000, 50000]

    train_siamese_network(csv_file, result_path, model_pretrained, num_pairs, save_pairs, convert_model)