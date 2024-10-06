import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.models import Model


def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def last_layer(encoded_l, encoded_r, lyr_name='cos'):
    if lyr_name == 'L1':
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        # drp_lyr = Dropout(0.25)(add_dens)
        # xx = Dense(128, activation='relu', bias_initializer=initialize_bias)(add_dens)
        # prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')

    elif lyr_name == 'L2':
        # Write L2 here
        L2_layer = Lambda(lambda tensors: (tensors[0] - tensors[1]) ** 2 / (tensors[0] + tensors[1]))
        L2_distance = L2_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        drp_lyr = Dropout(0.25)(add_dens)
        # xx =  Dense(128, activation='relu', bias_initializer=initialize_bias)(drp_lyr)
        # drp_lyr2 = Dropout(0.25)(xx)
        # x =  Dense(64, activation='relu', bias_initializer=initialize_bias)(xx)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(drp_lyr)

    else:
        # Add cosine similarity function
        cos_layer = Lambda(lambda tensors: K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                                           tf.keras.backend.l2_normalize(tensors[0]) * tf.keras.backend.l2_normalize(
            tensors[1]))
        cos_distance = cos_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(cos_distance)

    return prediction


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    siamese_model = Sequential()

    siamese_model.add(
        Conv1D(filters=256, kernel_size=50, strides=1, activation='relu',
               padding='same', input_shape=input_shape))  # lo strides di questo era impostato a 'strides=50'
    siamese_model.add(
        Conv1D(filters=128, kernel_size=10, strides=1, activation='relu',
               padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='sigmoid', padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    #######################################################################################
    # Ulterior 2 layers
    siamese_model.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid',
                             padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))

    siamese_model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='sigmoid',
                             padding='same'))
    siamese_model.add(MaxPooling1D(pool_size=2))
    #######################################################################################

    siamese_model.add(Flatten())

    encoded_l = siamese_model(left_input)
    encoded_r = siamese_model(right_input)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input],
                        outputs=last_layer(encoded_l, encoded_r, lyr_name='L2'),  # prediction and cosine_similarity
                        name="siamese-network")
    return siamese_net


def make_oneshot_task(genes_len, x_test, class_test_ind, N):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    X = x_test.values
    class_test_dic = class_test_ind

    list_N_samples = random.sample(list(class_test_dic.keys()), N)
    true_category = list_N_samples[0]
    out_ind = np.array([random.sample(class_test_dic[j], 2) for j in list_N_samples])
    indices = out_ind[:, 1]
    ex1 = out_ind[0, 0]

    # create one column of one sample
    test_image = np.asarray([X[ex1]] * N).reshape(N, genes_len, 1)
    support_set = X[indices].reshape(N, genes_len, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set, list_N_samples = shuffle(targets, test_image, support_set, list_N_samples)
    pairs = [test_image, support_set]

    return pairs, targets, true_category, list_N_samples


def test_oneshot(model, genes_len, x_test, class_test_ind, N, k, verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets, true_category, list_N_samples = make_oneshot_task(genes_len, x_test, class_test_ind, N)
        probs = model.predict(inputs, verbose=0)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct

def get_batch(batch_size, x_train, class_train_ind, genes_len, md='train'):
    """
    Create batch of n pairs, half same class, half different class
    """
    categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
    n_classes = len(class_train_ind.keys())

    pairs = [np.zeros((batch_size, genes_len, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size // 2:] = 1
    j = 0
    for i in range(batch_size):
        if j > n_classes - 1:
            categories = random.sample(list(class_train_ind.keys()), len(class_train_ind.keys()))
            j = 0

        category = categories[j]
        idx_1 = random.sample(class_train_ind[category], 1)[0]

        pairs[0][i, :, :] = x_train.values[idx_1].reshape(genes_len, 1)
        if i >= batch_size // 2:
            category_2 = category
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

        else:
            ind_pop = list(categories).index(category)
            copy_list = categories.copy()
            copy_list.pop(ind_pop)
            category_2 = random.sample(copy_list, 1)[0]
            idx_2 = random.sample(class_train_ind[category_2], 1)[0]
            pairs[1][i, :, :] = x_train.values[idx_2].reshape(genes_len, 1)

        j += 1
    return pairs, targets

def indices_save(dataset):
    # Creazione di una mappa dove la chiave è il tipo di attacco e
    # il valore è una lista degli indici delle righe
    class_map = {}
    for index, attack_type in dataset['type_attack'].items():
        if attack_type in class_map:
            class_map[attack_type].append(index)
        else:
            class_map[attack_type] = [index]
    return class_map


def siamese_network(risultati_siamese, dataset, input_shape, features_len, attack_type):
    siamese_model = get_siamese_model(input_shape)
    siamese_model.summary()

    optimizer = Adam(learning_rate=0.000005)
    siamese_model.compile(loss="binary_crossentropy", optimizer=optimizer)

    # Pre-Processing datasets for siamese network
    dataset = pd.concat([dataset, attack_type], axis=1)

    x_train, x_test = train_test_split(dataset, test_size=0.20, stratify=attack_type, shuffle=True)

    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    class_train_ind = indices_save(x_train)
    class_test_ind = indices_save(x_test)

    x_train = x_train.drop('type_attack', axis=1)
    x_test = x_test.drop('type_attack', axis=1)

    # Hyper parameters
    evaluate_every = 200  # interval for evaluating on one-shot tasks
    batch_size = 256
    n_iter = 1000000  # No. of training iterations
    N_way = len(class_test_ind.keys())  # how many classes for testing one-shot tasks
    n_val = 1000  # how many one-shot tasks to validate on
    best = -1

    print(f"\nNumero di classi train: {len(class_train_ind.keys())}")
    print(f"Numero di classi test: {len(class_test_ind.keys())}")

    print(f"\nNumero di features totali: {features_len}")
    print(f"Numero di pacchetti totali: {dataset.shape[0]}")

    print("\nStarting training process!")
    for i in range(1, n_iter + 1):
        (inputs, targets) = get_batch(batch_size, x_train, class_train_ind, features_len)
        loss = siamese_model.train_on_batch(inputs, targets)

        if i % 100 == 0:
            print(i)

        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Train Loss: {0}".format(loss))
            val_acc = test_oneshot(siamese_model, features_len, x_test, class_test_ind,
                                   N_way, n_val, verbose=True)
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                print(str(i))
                best = val_acc

                with open(risultati_siamese + "training_siamese.txt", 'a') as file:
                    file.write("Epoca: {0}, Current best: {1}, previous best: {2}".format(str(i), val_acc, best) + '\n')

                siamese_model.save(risultati_siamese + "siamese_model.keras")
                siamese_model.save(risultati_siamese + "siamese_model.h5")
                print("Model saved!")