from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from generate_pairs import *
from preprocessing import load_and_preprocess_data
from siamese_net import SiameseNet


# ==========================================================
#  Main training function
# ==========================================================
def train_siamese_network(csv_file):
    # Load and preprocess the dataset
    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.2)
    print("Data are preprocessed!")

    num_classes = len(label_encoder.classes_)
    print(num_classes)

    # Generation of pairs
    train_pairs, train_labels = generate_balanced_siamese_pairs(X_train, y_train, num_pairs=5, num_classes=num_classes)
    val_pairs, val_labels = generate_balanced_siamese_pairs(X_val, y_val, num_pairs=5, num_classes=num_classes)
    test_pairs, test_labels = generate_balanced_siamese_pairs(X_test, y_test, num_pairs=5, num_classes=num_classes)
    print("Pairs are generated!")

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

    # Evaluate the model on unseen test data
    test_loss, test_accuracy = siamese_model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Predictions on test set
    predictions = siamese_model.predict([test_pairs[:, 0], test_pairs[:, 1]]).ravel()
    # Threshold predictions to get binary labels
    predicted_labels = (predictions < 0.5).astype(int)

    print(test_labels)
    print(predicted_labels)

    # Evaluate the precision, recall, F1-score
    print(classification_report(test_labels, predicted_labels))

if __name__ == "__main__":
    csv_file = '../dataset_attacchi_bilanciato.csv'
    train_siamese_network(csv_file)