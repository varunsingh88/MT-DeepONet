"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: model script for FIsher problem
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
from keras.layers import Dense, Dropout, LayerNormalization
from keras import regularizers
from functools import partial

tf.keras.mixed_precision.set_global_policy('float32')


class Deep_Net_engine(tf.keras.Model):
    def __init__(self,
                 mlp_nodes_br,
                 act_br,
                 reg_br_lambda,
                 drpout_rate_br,
                 mlp_nodes_trk,
                 act_trk,
                 latent_dim):
        super().__init__()

        """
        Setting parameters for model
        """
        self.br1_hidden_nodes = mlp_nodes_br
        self.trk_hidden_nodes = mlp_nodes_trk
        self.latent_dim = latent_dim
        self.initializer_bias = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=123)
        self.initializer = tf.keras.initializers.GlorotNormal(seed=123)
        self.dropout_rate = drpout_rate_br
        self.regularizer = regularizers.L2(reg_br_lambda)
        self.br_act = tf.keras.layers.Activation(act_br)
        self.trk_act = tf.keras.layers.Activation(act_trk)
        self.mlp_lyr = partial(Dense, activation='linear', kernel_initializer=self.initializer,
                               kernel_regularizer=self.regularizer, bias_initializer=self.initializer_bias)
        self.mlp_trk_lyr = partial(Dense, activation='leaky_relu', kernel_initializer=self.initializer,
                                   bias_initializer=self.initializer_bias)

        """
        Branch1 layers
        """
        self.br1_l1 = self.mlp_lyr(self.br1_hidden_nodes[0], name='br1_l1')
        self.norm_br1_l1 = LayerNormalization()
        self.dropout_br1_l1 = Dropout(self.dropout_rate)
        self.br1_l2 = self.mlp_lyr(self.br1_hidden_nodes[1], name='br1_l2')
        self.norm_br1_l2 = LayerNormalization()
        self.dropout_br1_l2 = Dropout(self.dropout_rate)
        self.br1_l3 = self.mlp_lyr(self.latent_dim, name='br1_out')

        """
        Trunk layers
        """
        self.trk_l1 = self.mlp_trk_lyr(self.trk_hidden_nodes[0], name='trk_l1')
        self.trk_l2 = self.mlp_trk_lyr(self.trk_hidden_nodes[1], name='trk_l2')
        self.trk_l3 = self.mlp_trk_lyr(self.trk_hidden_nodes[2], name='trk_l3')
        self.trk_l4 = self.mlp_trk_lyr(self.latent_dim, activation='linear', name='trk_out')

    """
    Forward pass for DeepOnet source network
    """

    def call_don(self, x1, dom, training=None):
        br1_out = self.br1_l1(x1)
        br1_out = self.norm_br1_l1(br1_out, training=training)
        br1_out = self.br_act(br1_out)
        br1_out = self.dropout_br1_l1(br1_out, training=training)
        br1_out = self.br1_l2(br1_out)
        br1_out = self.norm_br1_l2(br1_out, training=training)
        br1_out = self.br_act(br1_out)
        br1_out = self.dropout_br1_l2(br1_out, training=training)
        br1_out = self.br1_l3(br1_out)

        trk_out = self.trk_l1(dom)
        trk_out = self.trk_l2(trk_out)
        trk_out = self.trk_l3(trk_out)
        trk_out = self.trk_l4(trk_out)

        return br1_out, trk_out
