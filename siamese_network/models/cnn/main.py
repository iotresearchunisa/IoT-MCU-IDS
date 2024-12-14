import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from models.siamese_net.utils.preprocessing import *
from cnn import *


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
def train_cnn_network(csv_file, path_results):

    ################################################################################################################
    print("\n================= [STEP 1.0] Load and preprocess the datasets =================")
    # Load and preprocess the datasets
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, label_encoder = load_and_preprocess_data(csv_file, test_size=0.999, val_size=0.2)
    print("Data are preprocessed!")
    ################################################################################################################


    ################################################################################################################
    print("\n================= [STEP 2.0] Training Phase =================")
    # Create Siamese model
    num_class = len(np.unique(y_val))

    cnn_model = CNN(num_class, (X_train.shape[1], 1)).get()

    # Define early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = cnn_model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            batch_size=256,
                            epochs=100,
                            callbacks=[early_stopping])

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
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predictions on test set
    y_pred_probs = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calcolo metriche generali
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='weighted')
    test_f1 = f1_score(y_test, y_pred, average='weighted')

    # Stampa dei risultati generali
    print("\nTest Results with CNN:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}\n")

    # Stampa del classification report dettagliato
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    ################################################################################################################


if __name__ == "__main__":
    csv_file = '../../datasets/TON_IoT/dataset.csv'
    result_path = "../../results/TON_IoT/cnn/07/"

    train_cnn_network(csv_file, result_path)