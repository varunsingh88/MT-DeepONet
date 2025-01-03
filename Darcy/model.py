"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: Source and Target models for Darcy flow problem
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Flatten, AveragePooling2D
from keras import regularizers
from functools import partial

tf.keras.mixed_precision.set_global_policy('float32')


class DeepONet(tf.keras.Model):
    def __init__(
            self, mlp_nodes_br, act_br_cnn, act_br_mlp, reg_br, reg_br_trnf, cnn_krnl_sz, cnn_filtr, cnn_stride,
            avg_pool, drpout_rate_br,
            trk_nodes, act_trk, reg_trk, latent_dim
    ):
        super().__init__()

        """
        Initializing network parameters
        """

        self.mlp_hidden_nodes = mlp_nodes_br
        self.latent_dim = latent_dim
        self.initializer_bias = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.01, seed=123
        )
        self.initializer = tf.keras.initializers.GlorotNormal(seed=123)
        self.dropout_rate = drpout_rate_br
        self.regularizer = regularizers.L2(reg_br)
        self.regularizer_trnf = regularizers.L2(reg_br_trnf)
        self.regularizer_trk = regularizers.L2(reg_trk)
        self.CNN_kernel_size = cnn_krnl_sz
        self.CNN_stride_len = cnn_stride
        self.CNN_avg_pool_size = avg_pool
        self.CNN_filter_size = cnn_filtr
        self.trk_hidden_nodes = trk_nodes

        self.mlp_lyr_br = partial(
            Dense,
            activation=act_br_mlp,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            bias_initializer=self.initializer_bias,
        )

        self.br_cnn_lyr = partial(
            Conv2D,
            kernel_size=self.CNN_kernel_size,
            strides=self.CNN_stride_len,
            padding="same",
            activation=act_br_cnn,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            bias_initializer=self.initializer_bias,
        )

        self.mlp_lyr_trk = partial(
            Dense,
            activation=act_trk,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer_trk,
            bias_initializer=self.initializer_bias,
        )

        """
        Branch CNN layers
        """
        self.br_l1 = self.br_cnn_lyr(self.CNN_filter_size[0], name='cnn_l1')
        self.br_l2 = AveragePooling2D(self.CNN_avg_pool_size, strides=2)
        self.br_l3 = self.br_cnn_lyr(self.CNN_filter_size[1], name='cnn_l2')
        self.br_l4 = AveragePooling2D(self.CNN_avg_pool_size, strides=2)
        self.br_l5 = self.br_cnn_lyr(self.CNN_filter_size[2], name='cnn_l3')
        self.br_l6 = AveragePooling2D(self.CNN_avg_pool_size, strides=2)
        self.br_l7 = self.br_cnn_lyr(self.CNN_filter_size[3], name='cnn_l4')
        self.br_l8 = AveragePooling2D(self.CNN_avg_pool_size, strides=2)
        self.br_l11 = Flatten()

        self.br_l12 = self.mlp_lyr_br(self.mlp_hidden_nodes[0], name='mlp1', kernel_regularizer=self.regularizer_trnf)
        self.dropout_l3 = Dropout(self.dropout_rate)
        self.br_l13 = self.mlp_lyr_br(self.mlp_hidden_nodes[1], name='mlp2', kernel_regularizer=self.regularizer_trnf)
        self.dropout_l4 = Dropout(self.dropout_rate)
        self.br_l14 = self.mlp_lyr_br(self.latent_dim, activation='linear', name='mlp3',
                                      kernel_regularizer=self.regularizer_trnf)

        """
        Trunk layers
        """
        self.trk_l1 = self.mlp_lyr_trk(self.trk_hidden_nodes[0], activation='leaky_relu', name='trk_l1')
        self.trk_l2 = self.mlp_lyr_trk(self.trk_hidden_nodes[1], activation='leaky_relu', name='trk_l2')
        self.trk_l3 = self.mlp_lyr_trk(self.trk_hidden_nodes[2], activation='leaky_relu', name='trk_l3')
        self.trk_l4 = self.mlp_lyr_trk(self.trk_hidden_nodes[3], activation='leaky_relu', name='trk_l4')
        self.trk_l5 = self.mlp_lyr_trk(self.latent_dim, activation='linear', name='trk_out')

    """
    Forward pass for DeepOnet source network
    """

    def call_don_source(self, x1, dom, training=None):
        br1_out = self.br_l1(x1)
        br1_out = self.br_l2(br1_out)
        br1_out = self.br_l3(br1_out)
        br1_out = self.br_l4(br1_out)
        br1_out = self.br_l5(br1_out)
        br1_out = self.br_l6(br1_out)
        br1_out = self.br_l7(br1_out)
        br1_out = self.br_l8(br1_out)
        br1_out = self.br_l11(br1_out)
        br1_out_mlp1 = self.br_l12(br1_out)
        br1_out_mlp1 = self.dropout_l3(br1_out_mlp1, training=training)
        br1_out_mlp2 = self.br_l13(br1_out_mlp1)
        br1_out_mlp2 = self.dropout_l4(br1_out_mlp2, training=training)
        br1_out_mlp3 = self.br_l14(br1_out_mlp2)

        trk_out = self.trk_l1(dom)
        trk_out = self.trk_l2(trk_out)
        trk_out = self.trk_l3(trk_out)
        trk_out = self.trk_l4(trk_out)
        trk_out = self.trk_l5(trk_out)

        return tf.reshape(br1_out_mlp3, [-1, 1, br1_out_mlp3.shape[1]]), trk_out, br1_out_mlp1, br1_out_mlp2

    def call_don_target(self, x1, dom, training=None):
        br1_out = self.br_l1(x1)
        br1_out = self.br_l2(br1_out)
        br1_out = self.br_l3(br1_out)
        br1_out = self.br_l4(br1_out)
        br1_out = self.br_l5(br1_out)
        br1_out = self.br_l6(br1_out)
        br1_out = self.br_l7(br1_out)
        br1_out = self.br_l8(br1_out)
        br1_out = self.br_l11(br1_out)
        br1_out_mlp1 = self.br_l12(br1_out)
        br1_out_mlp1 = self.dropout_l3(br1_out_mlp1, training=training)
        br1_out_mlp2 = self.br_l13(br1_out_mlp1)
        br1_out_mlp2 = self.dropout_l4(br1_out_mlp2, training=training)
        br1_out_mlp3 = self.br_l14(br1_out_mlp2)

        trk_out = self.trk_l1(dom)
        trk_out = self.trk_l2(trk_out)
        trk_out = self.trk_l3(trk_out)
        trk_out = self.trk_l4(trk_out)
        trk_out = self.trk_l5(trk_out)

        return tf.reshape(br1_out_mlp3, [-1, 1, br1_out_mlp3.shape[1]]), trk_out, br1_out_mlp1, br1_out_mlp2
