import matplotlib.pyplot as plt


def plot_snn_cnn_classifiers():
    splits = ['64%', '48%', '4%', '0.07%']
    x_values = range(len(splits))

    # SNN
    accuracies_snn_si_mio = [94.45, 94.63, 94.22, 83.87]
    accuracies_snn_tonIot = [91.29, 90.88, 89.31, 81.24]

    # CNN [COMPLETO]
    accuracies_cnn_mio    = [93.07, 92.44, 91.75, 77.90]
    accuracies_cnn_tonIot = [91.54, 91.15, 86.18, 70.43]

    # Classifiers
    accuracies_knn_mio = [92.28, 92.36, 91.53, 79.22]  # KNN
    accuracies_rf_mio  = [92.37, 92.21, 90.66, 76.58]  # Random Forest
    accuracies_svm_mio = [88.16, 88.13, 83.99, 76.35]  # SVM

    accuracies_knn_tonIot = [92.96, 92.94, 90.59, 79.77]  # KNN
    accuracies_rf_tonIot  = [93.50, 93.32, 91.38, 79.07]  # Random Forest
    accuracies_svm_tonIot = [84.78, 84.67, 82.15, 67.31]  # SVM



    ################# PLOT NO MQTT #################
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_snn_si_mio, marker='p', linestyle='-', color='orange', label='SNN')
    #for x, y in zip(x_values, accuracies_snn_si_mio):
        #plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='orange', fontsize=8)

    plt.plot(x_values, accuracies_cnn_mio, marker='+', linestyle='-', color='purple', label='CNN')

    plt.plot(x_values, accuracies_knn_mio, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(x_values, accuracies_rf_mio, marker='s', linestyle='-', color='green', label='RF')
    plt.plot(x_values, accuracies_svm_mio, marker='^', linestyle='-', color='red', label='SVM')

    plt.title('Accuracy at variation of Data Train (Few-Shot_IoT)') # 5 classi
    plt.xlabel('Train data percentage')
    plt.ylabel('Accuracy (%)')

    plt.xticks(x_values, splits)
    plt.ylim([60, 100])
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plot/few-shot_iot_split.png")
    plt.show()

    ################# PLOT SI MQTT #################
    plt.plot(x_values, accuracies_snn_tonIot, marker='p', linestyle='-', color='orange', label='SNN')
    #for x, y in zip(x_values, accuracies_snn_tonIot):
        #plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='orange', fontsize=8)

    plt.plot(x_values, accuracies_cnn_tonIot, marker='+', linestyle='-', color='purple', label='CNN')

    plt.plot(x_values, accuracies_knn_tonIot, marker='o', linestyle='-', color='blue', label='KNN')
    plt.plot(x_values, accuracies_rf_tonIot, marker='s', linestyle='-', color='green', label='RF')
    plt.plot(x_values, accuracies_svm_tonIot, marker='^', linestyle='-', color='red', label='SVM')

    plt.title('Accuracy at variation of Data Train (TON_IoT)')
    plt.xlabel('Train data percentage')
    plt.ylabel('Accuracy (%)')

    plt.xticks(x_values, splits)
    plt.ylim([60, 100])
    plt.grid(True)
    plt.legend()
    plt.savefig("results/plot/ton_iot_split.png")
    plt.show()


if __name__ == '__main__':
    plot_snn_cnn_classifiers()