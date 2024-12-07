import matplotlib.pyplot as plt

def plot_separate():
    splits = ['80-20 (0%)', '75-25 (-5%)', '70-30 (-10%)', '65-35 (-15%)', '60-40 (-20%)']
    x_values = range(len(splits))

    accuracies_no_mqtt = [99.15, 99.16, 98.99, 99.10, 99.16]
    accuracies_si_mqtt = [95.02, 95.01, 94.81, 94.86, 94.87]

    # Plot senza MQTT
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_no_mqtt, marker='o', linestyle='-', color='blue', label='no MQTT')
    for x, y in zip(x_values, accuracies_no_mqtt):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='blue', fontsize=8)

    plt.title('SNN accuracy at variation of split Train - Test (4 classes)')

    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')
    plt.xticks(x_values, splits)

    plt.ylim([90, 100])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot con MQTT
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_si_mqtt, marker='s', linestyle='-', color='green', label='MQTT')
    for x, y in zip(x_values, accuracies_si_mqtt):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='green', fontsize=8)

    plt.title('SNN accuracy at variation of split Train - Test (5 classes)')
    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')
    plt.xticks(x_values, splits)

    plt.ylim([90, 100])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot():
    splits = ['80-20 (0%)', '75-25 (-5%)', '70-30 (-10%)', '65-35 (-15%)', '60-40 (-20%)']
    x_values = range(len(splits))

    accuracies_no_mqtt = [99.15, 99.16, 98.99, 99.10, 99.16]
    accuracies_si_mqtt = [95.02, 95.01, 94.81, 94.86, 94.87]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, accuracies_no_mqtt, marker='o', linestyle='-', color='blue', label='no MQTT')
    for x, y in zip(x_values, accuracies_no_mqtt):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='blue', fontsize=8)

    plt.plot(x_values, accuracies_si_mqtt, marker='s', linestyle='-', color='green', label='MQTT')
    for x, y in zip(x_values, accuracies_si_mqtt):
        plt.text(x, y + 0.1, f'{y:.2f}', ha='center', va='bottom', color='green', fontsize=8)

    plt.title('SNN accuracy at variation of split Train - Test')
    plt.xlabel('Split Train - Test')
    plt.ylabel('Accuracy')

    plt.xticks(x_values, splits)
    plt.ylim([90, 100])
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot()
    plot_separate()