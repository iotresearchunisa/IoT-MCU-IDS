import matplotlib.pyplot as plt


def plot_values(x_values, y_values, title="Titolo del Grafico", x_label="Asse X", y_label="Asse Y"):
    if len(x_values) != len(y_values):
        raise ValueError("Gli array X e Y devono avere la stessa lunghezza.")

    # Crea il grafico
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')  # Disegna la linea con punti
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    x = [16, 20, 24, 28, 32, 36, 40]  # Valori asse X
    y = [0.9788, 0.9800, 0.9801, 0.9795, 0.9786, 0.9792, 0.9798]  # Valori asse Y

    titolo = "Validation"
    etichetta_x = "Val %"
    etichetta_y = "Acc"

    plot_values(x, y, title=titolo, x_label=etichetta_x, y_label=etichetta_y)
