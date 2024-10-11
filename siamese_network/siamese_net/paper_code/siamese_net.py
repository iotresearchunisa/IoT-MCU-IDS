from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Lambda, Conv1D, Flatten, MaxPooling1D
from keras.models import load_model


class SiameseNet:
    def __init__(self, input_shape, verbose=True):
        self.left_input = Input(input_shape)
        self.right_input = Input(input_shape)
        self.convnet = Sequential()

        lr = 0.0001 # metti 0.000001 per il transfer learning
        self.convnet.add(Conv1D(filters=256, kernel_size=50, strides=1, activation='relu', padding='same', input_shape=input_shape))
        self.convnet.add(Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', padding='same'))
        self.convnet.add(MaxPooling1D(pool_size=2))

        self.convnet.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='sigmoid', padding='same'))
        self.convnet.add(MaxPooling1D(pool_size=2))

        #######################################################################################
        # Ulterior 2 layers
        self.convnet.add(Conv1D(filters=64, kernel_size=3, strides=1, activation='sigmoid', padding='same'))
        self.convnet.add(MaxPooling1D(pool_size=2))

        self.convnet.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='sigmoid', padding='same'))
        self.convnet.add(MaxPooling1D(pool_size=2))
        #######################################################################################

        self.convnet.add(Flatten())

        # call the convnet Sequential model on each of the input tensors so params will be shared
        self.encoded_l = self.convnet(self.left_input)
        self.encoded_r = self.convnet(self.right_input)

        # layer to merge two encoded inputs with the l1 distance between them
        self.L1_layer = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)

        # call this layer on list of two input tensors.
        self.L1_distance = self.L1_layer([self.encoded_l, self.encoded_r])
        self.siamese_net = Model(inputs=[self.left_input, self.right_input], outputs=self.L1_distance)

        self.optimizer = Adam(lr)
        self.siamese_net.compile(loss=self.contrastive_loss, optimizer=self.optimizer, metrics=[self.accuracy])

        if verbose:
            print('Siamese Network Created:\n')
            self.siamese_net.summary()

    def get(self):
        return self.siamese_net

    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        y_true = K.cast(y_true, 'float32')
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    def load_saved_model(self, file_name):
        self.siamese_net = load_model(file_name, custom_objects={'contrastive_loss': self.contrastive_loss,
                                                                 'accuracy': self.accuracy,
                                                                 'euclidean_distance': self.euclidean_distance,
                                                                 'eucl_dist_output_shape': self.eucl_dist_output_shape})
        return self.siamese_net