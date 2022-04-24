import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate, Conv1D, Dense, Input, Permute

from layers import ConvBlock, DenseBlock
from tcn import TCN


def get_seq_model(
    conv_params,
    dense_params,
    window_size,
    n_ts_feature,
    n_local_feature,
    pred_step,
    use_exog=True,
):
    """
    model input shapes: ts_input: batch * window_size *  1 + n_ts_features
                        exog: batch * pred_step * n_local_feature
    :param conv_params:
    :param dense_params:
    :param window_size:
    :param n_ts_feature:
    :param n_local_feature:
    :param pred_step:
    :return:
    """
    ts_input = Input(shape=(window_size, 1 + n_ts_feature))

    """
    conv output
    """
    # window_size * channel
    conv_out = ConvBlock(**conv_params)(ts_input)
    # channel * window_size
    conv_out = Permute((2, 1))(conv_out)
    # channel * pred_step
    conv_out = Dense(
        pred_step,
        activation="relu",
        # kernel_regularizer=regularizers.l2(0.01),
        # bias_regularizer=regularizers.l2(0.01),
    )(conv_out)
    # pred_step * channel
    conv_out = Permute((2, 1))(conv_out)

    if not use_exog:
        # pred_step * 1
        conv_out = Dense(1, activation="linear")(conv_out)

        return tf.keras.models.Model(inputs=ts_input, outputs=conv_out)

    else:
        """
        concate conv output and exog data
        """
        exog = Input(shape=(pred_step, n_local_feature))

        # batch_size * horizon * channel + nlf
        concated = Concatenate(axis=2)([conv_out, exog])

        # concated = Conv1D(filters=8, padding="causal", kernel_size=4)(concated)

        """
        Final output
        """
        output = DenseBlock(**dense_params)(concated)

        return tf.keras.models.Model(inputs=[ts_input, exog], outputs=output)


def get_tcn_seq_model(
    tcn_params,
    dense_params,
    window_size,
    n_ts_feature,
    n_local_feature,
    pred_step,
    use_exog=True,
):
    """
    model input shapes: ts_input: batch * window_size *  1 + n_ts_features
                        exog: batch * pred_step * n_local_feature
    :param tcn_params:
    :param dense_params:
    :param window_size:
    :param n_ts_feature:
    :param n_local_feature:
    :param pred_step:
    :return:
    """
    ts_input = Input(shape=(window_size, 1 + n_ts_feature))

    """
    conv output
    """
    # window_size * channel
    conv_out = TCN(**tcn_params)(ts_input)
    # channel * window_size
    conv_out = Permute((2, 1))(conv_out)
    # channel * pred_step
    conv_out = Dense(
        pred_step,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    )(conv_out)
    # pred_step * channel
    conv_out = Permute((2, 1))(conv_out)

    if not use_exog:
        # pred_step * 1
        conv_out = Dense(1, activation="linear")(conv_out)

        return tf.keras.models.Model(inputs=ts_input, outputs=conv_out)

    else:
        """
        concate conv output and exog data
        """
        exog = Input(shape=(pred_step, n_local_feature))

        # batch_size * horizon * (channel + nlf)
        concated = Concatenate(axis=2)([conv_out, exog])

        """
        Final output
        """
        output = DenseBlock(**dense_params)(concated)

        return tf.keras.models.Model(inputs=[ts_input, exog], outputs=output)


def seq2seq_pred(model, ts_input, exog):
    output = model([ts_input, exog])
    return output.numpy().squeeze()
