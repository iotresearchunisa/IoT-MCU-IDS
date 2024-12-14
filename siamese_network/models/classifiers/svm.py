import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from utils.preprocessing import *


# ==========================================================
#  Main function to train and evaluate SVM with K-Fold
# ==========================================================
def train_svm(csv_file, k_folds):
    # Load and preprocess the dataset
    X_train, y_train, X_test, y_test, scaler, label_encoder = load_and_preprocess_data(csv_file, test_size=0.9992)

    # Initialize the SVM classifier
    svm_classifier = SVC()

    # Training metrics storage
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds)

    for train_index, val_index in kf.split(X_train):
        X_k_train, X_k_val = X_train[train_index], X_train[val_index]
        y_k_train, y_k_val = y_train[train_index], y_train[val_index]

        # Train the SVM classifier
        svm_classifier.fit(X_k_train, y_k_train)

        # Evaluate on the validation set
        y_val_pred = svm_classifier.predict(X_k_val)

        # Compute evaluation metrics on the validation set
        val_accuracy = accuracy_score(y_k_val, y_val_pred)
        val_precision = precision_score(y_k_val, y_val_pred, average='macro', zero_division=0)
        val_recall = recall_score(y_k_val, y_val_pred, average='macro', zero_division=0)
        val_f1 = f1_score(y_k_val, y_val_pred, average='macro', zero_division=0)

        # Store metrics
        metrics['accuracy'].append(val_accuracy)
        metrics['precision'].append(val_precision)
        metrics['recall'].append(val_recall)
        metrics['f1'].append(val_f1)

    # Average metrics over all folds
    avg_accuracy = np.mean(metrics['accuracy'])
    std_accuracy = np.std(metrics['accuracy'])

    avg_precision = np.mean(metrics['precision'])
    std_precision = np.std(metrics['precision'])

    avg_recall = np.mean(metrics['recall'])
    std_recall = np.std(metrics['recall'])

    avg_f1 = np.mean(metrics['f1'])
    std_f1 = np.std(metrics['f1'])

    # Stampa dei risultati medi e della deviazione standard
    print("\nTraining Results with KNN (K-Fold Cross-Validation):")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {std_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")

    # TESTING
    y_pred = svm_classifier.predict(X_test)

    # Compute evaluation metrics on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Print test results
    print("\nTest Results with SVM:")
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
    csv_file = '../../datasets/TON_IoT/dataset.csv'
    k_folds = 10

    train_svm(csv_file, k_folds)