import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use('Agg')  # Use a non-interactive backend for plotting


# Set random seeds for reproducibility
def set_seed(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ==========================================================
#  Check for GPU or CPU
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================================
#  Definition of the Siamese Network
# ==========================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


# ==========================================================
#  Contrastive loss function
# ==========================================================
def contrastive_loss(output1, output2, label, margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean(
        (1 - label) * torch.pow(euclidean_distance, 2) +
        label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    )
    return loss_contrastive


# ==========================================================
#  Dataset class for PyTorch (Training and Validation)
# ==========================================================
class NetworkFlowDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        # Create a dictionary mapping each label to its indices
        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        self.labels_set = list(np.unique(labels))

        # Balance positive and negative pairs
        self.num_samples = min(len(self.label_to_indices[0]), len(self.label_to_indices[1]))
        self.num_pairs = self.num_samples * 2  # Equal number of positive and negative pairs

        # To track generated pairs and prevent duplicates
        self.generated_pairs = set()

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        max_attempts = 1000
        attempts = 0
        sorted_indices = None
        label = None
        while attempts < max_attempts:
            if idx % 2 == 0:
                # Positive pair (same class)
                label = 0.0
                target_label = np.random.choice(self.labels_set)
                indices = np.random.choice(self.label_to_indices[target_label], size=2, replace=False)
                sorted_indices = tuple(sorted(indices))
            else:
                # Negative pair (different classes)
                label = 1.0
                label1, label2 = np.random.choice(self.labels_set, size=2, replace=False)
                x1_idx = np.random.choice(self.label_to_indices[label1])
                x2_idx = np.random.choice(self.label_to_indices[label2])
                sorted_indices = tuple(sorted((x1_idx, x2_idx)))

            if sorted_indices not in self.generated_pairs:
                self.generated_pairs.add(sorted_indices)
                break
            attempts += 1

        if attempts == max_attempts:
            print(f"Max attempts reached for index {idx}. Proceeding with duplicate pair.")
            # Even if the pair is duplicate, proceed by not adding it again
            # Optionally, you can choose to skip or handle it differently

        # At this point, sorted_indices and label are guaranteed to be assigned
        x1 = self.data[sorted_indices[0]].astype('float32')
        x2 = self.data[sorted_indices[1]].astype('float32')

        return torch.FloatTensor(x1), torch.FloatTensor(x2), torch.tensor(label, dtype=torch.float32)

    def check_duplicate_pairs(self, sample_size=10000):
        """
        Campiona un sottoinsieme di coppie per verificare i duplicati.
        :param sample_size: Numero di coppie da campionare.
        """
        pair_set = set()
        duplicates = 0
        sampled_indices = np.random.choice(len(self), size=min(sample_size, len(self)), replace=False)
        for idx in sampled_indices:
            x1, x2, label = self[idx]
            # Convert tensors to tuples for hashing
            x1_tuple = tuple(x1.numpy())
            x2_tuple = tuple(x2.numpy())
            # To treat (x1, x2) the same as (x2, x1), sort them
            if x1_tuple < x2_tuple:
                pair = (x1_tuple, x2_tuple)
            else:
                pair = (x2_tuple, x1_tuple)
            if pair in pair_set:
                duplicates += 1
            else:
                pair_set.add(pair)
        print(f"Sampled Pairs: {len(sampled_indices)}")
        print(f"Unique Pairs: {len(pair_set)}")
        print(f"Duplicate Pairs: {duplicates}")


# ==========================================================
#  Dataset class for testing phase
# ==========================================================
class SiameseNetworkTestDataset(Dataset):
    def __init__(self, data, labels, max_attempts=1000):
        self.data = data
        self.labels_array = labels
        self.max_attempts = max_attempts

        self.label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        self.labels_set = list(np.unique(labels))

        if len(self.labels_set) != 2:
            raise ValueError("Questo metodo supporta solo la classificazione binaria.")

        self.num_samples = min(len(self.label_to_indices[0]), len(self.label_to_indices[1]))
        self.num_pairs = self.num_samples * 2
        self.generated_pairs = set()

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        attempts = 0
        label = 0.0
        sorted_indices = None

        while attempts < self.max_attempts:
            if idx % 2 == 0:
                label = 0.0
                target_label = random.choice(self.labels_set)
                if len(self.label_to_indices[target_label]) < 2:
                    attempts += 1
                    continue
                indices = random.sample(list(self.label_to_indices[target_label]), 2)
            else:
                label = 1.0
                label1, label2 = random.sample(self.labels_set, 2)
                if len(self.label_to_indices[label1]) == 0 or len(self.label_to_indices[label2]) == 0:
                    attempts += 1
                    continue
                idx1 = random.choice(self.label_to_indices[label1])
                idx2 = random.choice(self.label_to_indices[label2])
                indices = [idx1, idx2]

            sorted_indices = tuple(sorted(indices))

            if sorted_indices not in self.generated_pairs:
                self.generated_pairs.add(sorted_indices)
                break

            attempts += 1

        if attempts == self.max_attempts:
            print(f"Max attempts reached for index {idx}. Procedendo con una coppia duplicata.")

        x1 = self.data[sorted_indices[0]].astype('float32')
        x2 = self.data[sorted_indices[1]].astype('float32')

        return torch.FloatTensor(x1), torch.FloatTensor(x2), torch.tensor(label, dtype=torch.float32)

    def check_duplicate_pairs(self, sample_size=10000):
        """
        Campiona un sottoinsieme di coppie per verificare i duplicati.
        :param sample_size: Numero di coppie da campionare.
        """
        pair_set = set()
        duplicates = 0
        sampled_indices = np.random.choice(len(self), size=min(sample_size, len(self)), replace=False)
        for idx in sampled_indices:
            x1, x2, label = self[idx]
            # Converti i tensori in tuple per il hashing
            x1_tuple = tuple(x1.numpy())
            x2_tuple = tuple(x2.numpy())
            # Tratta (x1, x2) come (x2, x1)
            if x1_tuple < x2_tuple:
                pair = (x1_tuple, x2_tuple)
            else:
                pair = (x2_tuple, x1_tuple)
            if pair in pair_set:
                duplicates += 1
            else:
                pair_set.add(pair)
        print(f"Sampled Pairs: {len(sampled_indices)}")
        print(f"Unique Pairs: {len(pair_set)}")
        print(f"Duplicate Pairs: {duplicates}")


# ==========================================================
#  Preprocessing function
# ==========================================================
def preprocess_dataset(df, train=True, preprocessor=None):
    # Separate features and target
    X = df.drop(columns=['type_attack'])
    y = df['type_attack']

    # Identify categorical and numerical columns
    categorical_cols = ['Protocol_Type']
    numerical_cols = X.columns.difference(categorical_cols)

    # Fill missing values in numerical columns
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

    # Fill missing values in categorical columns
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

    # Convert 'type_attack' to binary labels (0 for benign, 1 for attack)
    y = y.apply(lambda x: 0 if x == 'benign' else 1).values

    if train:
        # Define the ColumnTransformer with OneHotEncoder and RobustScaler
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ])
        # Fit and transform the training data
        X = preprocessor.fit_transform(X)
    else:
        # Transform the data using the same preprocessor
        X = preprocessor.transform(X)

    # Convert the processed data to a numpy array of type float32
    X = X.astype('float32')

    return X, y, preprocessor


# ==========================================================
#  Load and preprocess dataset
# ==========================================================
def load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.2):
    df = pd.read_csv(csv_file, sep=";")

    # Split into train+val/test before preprocessing
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['type_attack'], random_state=42)

    # Split train_val_df into train/val
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, stratify=train_val_df['type_attack'],
                                        random_state=42)

    # Preprocess the datasets
    X_train, y_train, preprocessor = preprocess_dataset(train_df, train=True)
    X_val, y_val, _ = preprocess_dataset(val_df, train=False, preprocessor=preprocessor)
    X_test, y_test, _ = preprocess_dataset(test_df, train=False, preprocessor=preprocessor)

    # Check for overlap between train, val, and test sets
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    test_indices = set(test_df.index)

    print(f"Training Set Size: {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Test Set Size: {len(test_df)}")

    assert train_indices.isdisjoint(val_indices), "Train and validation sets overlap!"
    assert train_indices.isdisjoint(test_indices), "Train and test sets overlap!"
    assert val_indices.isdisjoint(test_indices), "Validation and test sets overlap!"

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================================
#  Analyze distances and find optimal threshold
# ==========================================================
def analyze_distances(siamese_net, dataloader):
    siamese_net.eval()
    distances = []
    labels = []

    with torch.no_grad():
        for x1, x2, label in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            label = label.to(device).view(-1)  # Ensure label is 1D
            output1 = siamese_net(x1)
            output2 = siamese_net(x2)
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            distances.extend(euclidean_distance.cpu().numpy())
            labels.extend(label.cpu().numpy())

    distances = np.array(distances)
    labels = np.array(labels)

    # Flatten labels if necessary
    if labels.ndim > 1:
        labels = labels.flatten()

    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(distances[labels == 0], bins=50, alpha=0.5, label='Similar Pairs')
    plt.hist(distances[labels == 1], bins=50, alpha=0.5, label='Dissimilar Pairs')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.legend()
    plt.savefig('distance_distribution.png')
    plt.close()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('roc_curve.png')
    plt.close()

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal Threshold: {optimal_threshold}')

    return optimal_threshold


# ==========================================================
#  Test the model
# ==========================================================
def test_model(siamese_net, test_dataloader, threshold):
    siamese_net.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for x1, x2, label in test_dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            label = label.to(device).view(-1)

            output1 = siamese_net(x1)
            output2 = siamese_net(x2)

            # Calcolo della distanza euclidea
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > threshold).float()

            # Aggiunge etichette e predizioni al volo
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Converti le liste in numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calcolo delle metriche finali
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    # Stampa i risultati
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # Calcolo e visualizzazione della matrice di confusione
    cmatrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cmatrix)

    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Similar', 'Dissimilar'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, precision, recall, f1


# ==========================================================
#  Main training function
# ==========================================================
def train_siamese_network(csv_file):
    # Load and preprocess the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(csv_file, test_size=0.2, val_size=0.5)

    # Create DataLoaders for training and validation
    train_dataset = NetworkFlowDataset(X_train, y_train)
    val_dataset = NetworkFlowDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Verifica delle coppie duplicate nel Training e Validation Set
    print("\nDuplicati TRAIN:")
    train_dataset.check_duplicate_pairs(sample_size=10000)

    print("\nDuplicati VAL:")
    val_dataset.check_duplicate_pairs(sample_size=10000)

    # Create the Siamese Network model
    input_size = X_train.shape[1]
    siamese_net = SiameseNetwork(input_size).to(device)

    # Define optimizer and scheduler
    optimizer = optim.Adam(siamese_net.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    num_epochs = 100
    patience = 5 # Per Early Stopping
    best_val_loss = np.inf
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    siamese_net.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for x1, x2, label in train_dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            label = label.to(device).view(-1)  # Ensure label is 1D

            optimizer.zero_grad()
            output1 = siamese_net(x1)
            output2 = siamese_net(x2)
            loss = contrastive_loss(output1, output2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            euclidean_distance = nn.functional.pairwise_distance(output1, output2)
            predictions = (euclidean_distance > 1.0).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)

        train_accuracy = correct / total
        train_losses.append(running_loss / len(train_dataloader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        siamese_net.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x1, x2, label in val_dataloader:
                x1, x2 = x1.to(device), x2.to(device)
                label = label.to(device).view(-1)  # Ensure label is 1D

                output1 = siamese_net(x1)
                output2 = siamese_net(x2)
                loss = contrastive_loss(output1, output2, label)
                val_running_loss += loss.item()

                # Calculate validation accuracy
                euclidean_distance = nn.functional.pairwise_distance(output1, output2)
                predictions = (euclidean_distance > 1.0).float()
                val_correct += (predictions == label).sum().item()
                val_total += label.size(0)

        val_accuracy = val_correct / val_total
        val_losses.append(val_running_loss / len(val_dataloader))
        val_accuracies.append(val_accuracy)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {running_loss / len(train_dataloader):.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_running_loss / len(val_dataloader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")

        # Early Stopping
        current_val_loss = val_running_loss / len(val_dataloader)
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            epochs_no_improve = 0

            # Salva il modello migliore
            torch.save(siamese_net.state_dict(), 'best_siamese_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    print("Training completed.")

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('accuracy_plot.png')
    plt.close()

    # Carica il modello migliore
    siamese_net.load_state_dict(torch.load('best_siamese_model.pth'))

    # Analyze distances and find optimal threshold using validation data
    optimal_threshold = analyze_distances(siamese_net, val_dataloader)

    # Prepare test data
    test_dataset = SiameseNetworkTestDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\nDuplicati TEST:")
    test_dataset.check_duplicate_pairs(sample_size=10000)

    # Test the model
    test_model(siamese_net, test_dataloader, optimal_threshold)

if __name__ == "__main__":
    csv_file = '../dataset/dataset_bilanciato.csv'
    train_siamese_network(csv_file)