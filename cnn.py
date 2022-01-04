import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D, Dense, GlobalMaxPooling1D, MaxPooling1D, Dropout, Embedding, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch

class CNN:
    def __init__(self, vocab_size, batch_size=256, epochs=20, embedding_dim=100, seq_len=30):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.seq_len = seq_len
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=0)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=0.000001, verbose=0)
        self.model = self.build_model(vocab_size)

    def build_model(self, vocab_size):
        inputs = Input(shape=(self.seq_len,), dtype=np.int32)
        embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, input_length=self.seq_len, trainable=True)
        model = embedding(inputs)
        model = Dropout(0.2)(model)

        # conv block 1
        model = Conv1D(128, 3, activation='relu')(model)
        model = Conv1D(128, 3, activation='relu')(model)
        model = MaxPooling1D()(model)

        # conv block 2
        model = Conv1D(64, 3, activation='relu')(model)
        model = Conv1D(64, 3, activation='relu')(model)
        model = Dropout(0.2)(model)

        # conv block 3
        model = Conv1D(64, 3, activation='relu')(model)
        model = Conv1D(32, 3, activation='relu')(model)
        model = GlobalMaxPooling1D()(model)

        # FC
        model = BatchNormalization()(model)
        model = Dense(512, activation='relu')(model)
        model = Dense(128, activation='relu')(model)
        model = Dropout(0.2)(model)

        # output
        outputs = Dense(6, activation='softmax')(model)
        return tf.keras.Model(inputs, outputs, name='cnn')

    def summary(self):
        return self.model.summary()

    def train(self, xtrain, xtest, ytrain, ytest):
        opt = Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', 'AUC'])
        history = self.model.fit(xtrain, ytrain, batch_size=self.batch_size, epochs=self.epochs, validation_data=(xtest, ytest), callbacks=[self.early_stop, self.reduce_lr], verbose=1)
        return history
