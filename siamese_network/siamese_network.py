import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# ==========================================================
#  Controllo dispositivo: GPU o CPU
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")


# ==========================================================
#  Definizione della rete siamese
# ==========================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x1, x2):
        output1 = self.fc(x1)
        output2 = self.fc(x2)
        return output1, output2


# ==========================================================
#  Funzione per il contrastive loss
# ==========================================================
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                      label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss


# ==========================================================
#  Classe Dataset per PyTorch
# ==========================================================
class NetworkFlowDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Estrarre una coppia di campioni con la stessa label
        x1 = self.data[index]
        label = self.labels[index]
        index2 = np.random.choice(np.where(self.labels == label)[0])
        x2 = self.data[index2]
        return torch.FloatTensor(x1), torch.FloatTensor(x2), torch.FloatTensor([label])


# ==========================================================
#  Classe Dataset per la fase di test
# ==========================================================
class SiameseNetworkTestDataset(Dataset):
    def __init__(self, data, labels):
        self.pairs = []
        self.labels = []

        # Crea coppie sia della stessa classe che di classi diverse
        self.indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        min_class_samples = min([len(self.indices[label]) for label in self.indices])

        for idx in range(min_class_samples):
            # Coppie della stessa classe
            for label in self.indices:
                idx_a = self.indices[label][idx]
                idx_b = self.indices[label][(idx+1)%len(self.indices[label])]
                self.pairs.append([data[idx_a], data[idx_b]])
                self.labels.append(0) # Etichetta 0 per stessa classe

            # Coppie di classi diverse
            labels_list = list(self.indices.keys())
            label_a, label_b = labels_list[0], labels_list[1]
            idx_a, idx_b = self.indices[label_a][idx], self.indices[label_b][idx]
            self.pairs.append([data[idx_a], data[idx_b]])
            self.labels.append(1) # Etichetta 1 per classi diverse

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        x1, x2 = self.pairs[index]
        label = self.labels[index]
        return torch.FloatTensor(x1), torch.FloatTensor(x2), torch.FloatTensor([label])


# ==========================================================
#  Funzione per il preprocessing del dataset
# ==========================================================
def preprocess_dataset(df, train=True, scaler=None):
    # Codificare la colonna Protocol_Type
    le = LabelEncoder()
    df['Protocol_Type'] = le.fit_transform(df['Protocol_Type'])

    # Normalizzare le feature numeriche
    numeric_features = df.select_dtypes(include=['float64']).columns
    if train:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
    else:
        df[numeric_features] = scaler.transform(df[numeric_features])

    # Convertire la colonna type_attack in etichette binarie (0 per benign, 1 per attack)
    df['type_attack'] = df['type_attack'].apply(lambda x: 0 if x == 'benign' else 1)

    # Separare le feature dal target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack'].values

    return X.values, y, scaler


# ==========================================================
#  Caricamento e preprocessing del dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Dividsione in train/test prima di applicare il preprocessing
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], random_state=42)

    # Preprocessa il training set e applica lo scaler
    X_train, y_train, scaler = preprocess_dataset(train_df, train=True)

    # Preprocessa il test set usando lo stesso scaler
    X_test, y_test, _ = preprocess_dataset(test_df, train=False, scaler=scaler)

    return X_train, X_test, y_train, y_test


# ==========================================================
#  Funzione per il test del modello
# ==========================================================
def test_model(siamese_net, test_dataloader, threshold):
    siamese_net.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for x1, x2, label in test_dataloader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            output1, output2 = siamese_net(x1, x2)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > threshold).float()
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calcolo delle metriche
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")


# ==========================================================
#  Funzione principale per l'addestramento
# ==========================================================
def train_siamese_network(csv_file):
    # Caricamento e preprocessing del dataset
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_file)

    # Creazione del DataLoader per il training
    train_dataset = NetworkFlowDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Creazione del modello siamese
    input_size = X_train.shape[1]
    siamese_net = SiameseNetwork(input_size).to(device)

    # Definizione dell'ottimizzatore e scheduler
    optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Addestramento del modello
    num_epochs = 20
    train_accuracies = []
    siamese_net.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x1, x2, label) in enumerate(train_dataloader):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = siamese_net(x1, x2)
            loss = contrastive_loss(output1, output2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calcolo dell'accuracy sui batch durante l'addestramento
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > 0.5).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)

        accuracy = correct / total
        train_accuracies.append(accuracy)
        scheduler.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader)}")

    print("Addestramento completato.")

    # Plot dell'accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy nel tempo')
    plt.legend()
    plt.grid()
    plt.show()

    # Fase di test
    test_dataset = SiameseNetworkTestDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    threshold = 0.41

    # Test del modello
    test_model(siamese_net, test_dataloader, threshold)


if __name__ == "__main__":
    csv_file = '../dataset/dataset_bilanciato.csv'
    train_siamese_network(csv_file)
