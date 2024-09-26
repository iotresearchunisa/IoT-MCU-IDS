import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# ==========================================================
#  Function to plot both train and validation loss
# ==========================================================
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')  # You can change the path if needed
    plt.close()


# ==========================================================
#  Function to plot both train and validation accuracy
# ==========================================================
def plot_accuracy(train_accuracies, val_accuracies):
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


# ==========================================================
#  Function to plot confusion matrix
# ==========================================================
def plot_cmatrix(cmatrix):
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=['Similar', 'Dissimilar'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()


# ==========================================================
#  Function to plot ROC curve
# ==========================================================
def plot_rocurve(fpr, tpr, roc_auc):
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


# ==========================================================
#  Function to plot Histograms
# ==========================================================
def plot_histograms(distances, labels):
    plt.figure(figsize=(10, 5))
    plt.hist(distances[labels == 0], bins=50, alpha=0.5, label='Similar Pairs')
    plt.hist(distances[labels == 1], bins=50, alpha=0.5, label='Dissimilar Pairs')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances')
    plt.legend()
    plt.savefig('distance_distribution.png')
    plt.close()