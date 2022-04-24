import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
)


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=(
            8,
            8,
        ),
        filter_size=(
            8,
            8,
        ),
        dropout=0,
        batch_norm=False,
        padding="causal",
        dense_size=(8,),
        activation="relu",
        l2=0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.padding = padding
        self.dense_size = dense_size
        self.activation = activation
        self.l2 = l2
        self.layers = []

    def build(self, input_shape):

        for i in range(len(self.kernel_size)):
            self.layers.append(
                Conv1D(
                    kernel_size=self.kernel_size[i],
                    filters=self.filter_size[i],
                    padding=self.padding,
                    activation="linear",
                    kernel_regularizer=regularizers.l2(self.l2)
                )
            )

            if self.batch_norm:
                self.layers.append(BatchNormalization())

            if i < len(self.kernel_size) - 1:
                self.layers.append(Activation(self.activation))

            if self.dropout > 0:
                self.layers.append(Dropout(rate=self.dropout))

    def call(self, x, training):
        for layer in self.layers:
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size=(8,), activation="relu", dropout=0.0, l2=0.0):
        self.hidden_size = hidden_size
        self.activation = activation
        self.dropout = dropout
        self.l2 = l2
        self.layers = []
        super().__init__()

    def build(self, input_shape):
        for i, s in enumerate(self.hidden_size):
            if self.dropout > 0.:
                self.layers.append(Dropout(rate=self.dropout))

            self.layers.append(
                Dense(
                    s,
                    kernel_regularizer=regularizers.l2(self.l2),
                    activation=self.activation
                    if i < len(self.hidden_size) - 1
                    else "linear",
                )
            )

    def call(self, x, training):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

