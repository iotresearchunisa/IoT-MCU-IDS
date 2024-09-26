import tensorflow as tf
from preprocessing import *
from siamese_net import SiameseNet
from generate_pairs import *
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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
def train_siamese_network(csv_file):
    # Load and preprocess the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.2)
    print("Data are preprocessed!")

    # Generation of pairs
    train_pairs, train_labels = generate_balanced_siamese_pairs(X_train, y_train, 60000)
    val_pairs, val_labels = generate_balanced_siamese_pairs(X_val, y_val, 60000)
    test_pairs, test_labels = generate_balanced_siamese_pairs(X_test, y_test, 20000)
    print("Pairs are preprocessed!")

    # Create Siamese model
    siamese_model = (SiameseNet(input_shape=(X_train.shape[1], 1))).get()

    # Define early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = siamese_model.fit([train_pairs[:, 0], train_pairs[:, 1]],
                                train_labels,
                                validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
                                batch_size=64,
                                epochs=1000,
                                callbacks=[early_stopping])

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Test the model on unseen test data
    test_loss, test_accuracy = siamese_model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predictions on test set (optional)
    predictions = siamese_model.predict([test_pairs[:, 0], test_pairs[:, 1]])
    # Example: You could threshold predictions if you're treating this as binary classification
    predicted_labels = (predictions > 0.5).astype(int)

    # Evaluate the precision, recall, F1-score (if needed)
    from sklearn.metrics import classification_report
    print(classification_report(test_labels, predicted_labels))


if __name__ == "__main__":
    csv_file = 'dataset_bilanciato.csv'
    train_siamese_network(csv_file)
