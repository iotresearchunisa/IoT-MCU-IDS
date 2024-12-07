import matplotlib.pyplot as plt

def plot_separate(flag):
    splits = ['80-20 (0%)', '75-25 (-5%)', '70-30 (-10%)', '65-35 (-15%)', '60-40 (-20%)']
    x_values = range(len(splits))

    if flag:
        # Senza MQTT
        accuracies_knn = [99.33, 99.26, 99.23, 99.27, 99.27]  # KNN
        accuracies_rf = [99.21, 99.15, 99.19, 99.25, 99.20]  # Random Forest
        accuracies_svm = [96.59, 95.36, 95.37, 95.42, 95.34]  # SVM
    else:
        # Con MQTT
        accuracies_knn = [92.49, 92.40, 92.39, 92.33, 92.40]  # KNN
        accuracies_rf = [92.29, 92.38, 92.28, 92.22, 92.10]  # Random Forest
        accuracies_svm = [88.85, 88.77, 88.21, 88.51, 88.35]  # SVM

    # Plot per KNN
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_knn, marker='o', linestyle='-', color='blue', label='KNN')
    for x, y in zip(x_values, accuracies_knn):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='blue', fontsize=8)

    if flag:
        plt.title('KNN accuracy at variation of split Train - Test (4 classes)')
    else:
        plt.title('KNN accuracy at variation of split Train - Test (5 classes)')

    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')
    plt.xticks(x_values, splits)

    if flag:
        plt.ylim([90, 100])
    else:
        plt.ylim([80, 100])

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot per Random Forest
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_rf, marker='s', linestyle='-', color='green', label='Random Forest')
    for x, y in zip(x_values, accuracies_rf):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='green', fontsize=8)

    if flag:
        plt.title('Random Forest accuracy at variation of split Train - Test (4 classes)')
    else:
        plt.title('Random Forest accuracy at variation of split Train - Test (5 classes)')

    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')
    plt.xticks(x_values, splits)

    if flag:
        plt.ylim([90, 100])
    else:
        plt.ylim([80, 100])

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot per SVM
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_svm, marker='^', linestyle='-', color='red', label='SVM')
    for x, y in zip(x_values, accuracies_svm):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='red', fontsize=8)

    if flag:
        plt.title('SVM accuracy at variation of split Train - Test (4 classes)')
    else:
        plt.title('SVM accuracy at variation of split Train - Test (5 classes)')

    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')
    plt.xticks(x_values, splits)

    if flag:
        plt.ylim([90, 100])
    else:
        plt.ylim([80, 100])

    plt.grid(True)
    plt.legend()
    #plt.legend(loc='upper right', bbox_to_anchor=(0.99, 0.98), borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def plot(flag):
    splits = ['80-20 (0%)', '75-25 (-5%)', '70-30 (-10%)', '65-35 (-15%)', '60-40 (-20%)']
    x_values = range(len(splits))

    if flag:
        # Senza MQTT
        accuracies_knn = [99.33, 99.26, 99.23, 99.27, 99.27] # KNN
        accuracies_rf = [99.21, 99.15, 99.19, 99.25, 99.20]  # Random Forest
        accuracies_svm = [96.59, 95.36, 95.37, 95.42, 95.34] # SVM
    else:
        # Con MQTT
        accuracies_knn = [92.49, 92.40, 92.39, 92.33, 92.40] # KNN
        accuracies_rf = [92.29, 92.38, 92.28, 92.22, 92.10]  # Random Forest
        accuracies_svm = [88.85, 88.77, 88.21, 88.51, 88.35] # SVM

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_knn, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(x_values, accuracies_rf, marker='s', linestyle='-', color='green', label='Random Forest')
    plt.plot(x_values, accuracies_svm, marker='^', linestyle='-', color='red', label='SVM')

    if flag:
        plt.title('Accuracy at variation of split Train - Test (4 classes)')
    else:
        plt.title('Accuracy at variation of split Train - Test (5 classes)')
    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')

    plt.xticks(x_values, splits)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    senza_mqtt = False
    plot(senza_mqtt)
    plot_separate(senza_mqtt)