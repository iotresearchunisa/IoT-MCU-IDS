from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import *
import pickle


# ==========================================================
#  Main function to train and evaluate KNN
# ==========================================================
def train_knn(csv_file, model_filename, train):
    if train:
        # PRE-PROCESSING
        X_train, y_train, X_test, y_test, scaler, label_encoder = load_and_preprocess_data(csv_file, test_size=0.40)

        # TRAINING
        knn_classifier = KNeighborsClassifier(n_neighbors=5)

        # Train the KNN classifier
        knn_classifier.fit(X_train, y_train)

        # Salvataggio del modello addestrato e del label encoder
        with open(model_filename, 'wb') as file:
            pickle.dump((knn_classifier, scaler, label_encoder), file)
        print(f"Modello e pre-processing salvati come '{model_filename}'.")

        # Evaluate on the training set
        y_train_pred = knn_classifier.predict(X_train)

        # Compute evaluation metrics on the training set
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

        # Print training results
        print("\nTraining Results with KNN:")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Training Precision: {train_precision:.4f}")
        print(f"Training Recall: {train_recall:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")
    else:
        with open(model_filename, 'rb') as file:
            knn_classifier, scaler, label_encoder = pickle.load(file)
        print(f"Modello caricato da '{model_filename}'.")

        df = pd.read_csv(csv_file, sep=";")

        # PRE-PROCESSING
        X_test, y_test, _, _ = preprocess_dataset(df, train=False, scaler=scaler, label_encoder=label_encoder)

    # TESTING
    y_pred = knn_classifier.predict(X_test)

    # Compute evaluation metrics on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Print test results
    print("\nTest Results with KNN:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Compute and display the confusion matrix on the test set
    cmatrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix on Test Set:")
    print(cmatrix)

    # Display the classification report
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    csv_file = '../datasets/mio/dataset_attacchi_con_MQTT_bilanciato.csv'
    model_filename = 'results/train_reduced/mio/con_mqtt/_20/knn/knn.pkl'
    train = True

    train_knn(csv_file, model_filename, train)