from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from preprocessing import *
import pickle


# ==========================================================
#  Funzione principale per addestrare e valutare il Random Forest
# ==========================================================
def train_svm(csv_file, model_filename, train):
    if train:
        # Carica e preprocessa il datasets
        X_train, y_train, X_test, y_test, scaler, label_encoder = load_and_preprocess_data(csv_file, test_size=0.40)

        # Inizializza il classificatore SVM
        svm_classifier = SVC()

        # Addestra il classificatore SVM
        svm_classifier.fit(X_train, y_train)

        # Salvataggio del modello addestrato e del label encoder
        with open(model_filename, 'wb') as file:
            pickle.dump((svm_classifier, scaler, label_encoder), file)
        print(f"Modello e pre-processing salvati come '{model_filename}'.")

        # Valuta sul set di training
        y_train_pred = svm_classifier.predict(X_train)

        # Calcola le metriche di valutazione sul training
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

        # Stampa i risultati sul training
        print("\nRisultati sul Training:")
        print(f"Accuracy del Training: {train_accuracy:.4f}")
        print(f"Precisione del Training: {train_precision:.4f}")
        print(f"Recall del Training: {train_recall:.4f}")
        print(f"F1 Score del Training: {train_f1:.4f}")
    else:
        with open(model_filename, 'rb') as file:
            svm_classifier, scaler, label_encoder = pickle.load(file)
        print(f"Modello caricato da '{model_filename}'.")

        df = pd.read_csv(csv_file, sep=";")

        # PRE-PROCESSING
        X_test, y_test, _, _ = preprocess_dataset(df, train=False, scaler=scaler, label_encoder=label_encoder)

    # Valuta sul set di test
    y_pred = svm_classifier.predict(X_test)

    # Calcola le metriche di valutazione sul test
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # Stampa i risultati sul test
    print("\nRisultati sul Test:")
    print(f"Accuracy del Test: {test_accuracy:.4f}")
    print(f"Precisione del Test: {test_precision:.4f}")
    print(f"Recall del Test: {test_recall:.4f}")
    print(f"F1 Score del Test: {test_f1:.4f}")

    # Calcola e visualizza la matrice di confusione sul test
    cmatrix = confusion_matrix(y_test, y_pred)
    print("\nMatrice di Confusione sul Test:")
    print(cmatrix)

    # Visualizza il report di classificazione
    class_names = label_encoder.classes_
    print("\nReport di Classificazione:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

if __name__ == "__main__":
    csv_file = '../datasets/mio/dataset_attacchi_con_MQTT_bilanciato.csv'
    model_filename = 'results/train_reduced/mio/con_mqtt/_20/svm/svm.pkl'
    train = True

    train_svm(csv_file, model_filename, train)
