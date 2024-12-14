from keras.src.layers import MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class CNN:
    def __init__(self, num_class, input_shape, verbose=True):
        self.model = Sequential()

        # Blocco 1
        self.model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.3))

        # Blocco 2
        self.model.add(Conv1D(128, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=2))

        # Blocco 3
        self.model.add(Conv1D(256, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=2))

        # Blocco 4
        self.model.add(Conv1D(256, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(pool_size=2))

        # Flatten e Dense
        self.model.add(Flatten())

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))

        # Output finale
        self.model.add(Dense(num_class, activation='softmax'))

        lr = 0.0001
        optimizer = Adam(lr=lr)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        if verbose:
            print('CNN Created:\n')
            self.model.summary()

    def get(self):
        return self.model