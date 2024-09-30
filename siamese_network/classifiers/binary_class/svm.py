from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.preprocessing import load_and_preprocess_data


# ==========================================================
#  Funzione principale per addestrare e valutare SVM
# ==========================================================
def train_svm(csv_file):
    # Carica e preprocessa il dataset
    X_train, y_train, X_test, y_test = load_and_preprocess_data(csv_file, test_size=0.2)

    # Inizializza il classificatore SVM
    svm_classifier = SVC(random_state=42)

    # Addestra il classificatore SVM
    svm_classifier.fit(X_train, y_train)

    # Valuta sul set di training
    y_train_pred = svm_classifier.predict(X_train)

    # Calcola le metriche di valutazione sul training
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, pos_label=1, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1, zero_division=0)

    # Stampa i risultati sul training
    print("\nRisultati sul Training con SVM:")
    print(f"Accuracy del Training: {train_accuracy:.4f}")
    print(f"Precisione del Training: {train_precision:.4f}")
    print(f"Recall del Training: {train_recall:.4f}")
    print(f"F1 Score del Training: {train_f1:.4f}")

    # Valuta sul set di test
    y_pred = svm_classifier.predict(X_test)

    # Calcola le metriche di valutazione sul test
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Stampa i risultati sul test
    print("\nRisultati sul Test con SVM:")
    print(f"Accuracy del Test: {test_accuracy:.4f}")
    print(f"Precisione del Test: {test_precision:.4f}")
    print(f"Recall del Test: {test_recall:.4f}")
    print(f"F1 Score del Test: {test_f1:.4f}")

    # Calcola e visualizza la matrice di confusione sul test
    cmatrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("\nMatrice di Confusione sul Test:")
    print(cmatrix)


if __name__ == "__main__":
    csv_file = '../../dataset_bilanciato.csv'
    train_svm(csv_file)
