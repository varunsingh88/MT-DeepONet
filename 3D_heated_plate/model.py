"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: model script for heat transfer in 3D plate
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, BatchNormalization
from keras import regularizers
from functools import partial

tf.keras.mixed_precision.set_global_policy("float32")


class Deep_Net_engine(tf.keras.Model):
    def __init__(
            self,
            mlp_nodes_br,
            act_br,
            reg_br_lambda,
            drpout_rate_br,
            mlp_nodes_trk,
            act_trk,
            latent_dim,
            adaptive_act,
            residual_conn
    ):
        super().__init__()
        """
            Initializing network parameters
        """

        self.mlp_nodes_br = mlp_nodes_br
        self.latent_dim = latent_dim
        self.initializer_bias = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.001, seed=123
        )
        # self.initializer = tf.keras.initializers.GlorotNormal(seed=123)
        self.initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.001, seed=123
        )
        self.dropout_rate = drpout_rate_br
        self.regularizer = regularizers.L2(reg_br_lambda)
        self.mlp_nodes_trk = mlp_nodes_trk
        self.residual_conn = residual_conn
        self.br_act = tf.keras.layers.Activation(act_br)
        self.trk_act = tf.keras.layers.Activation(act_trk)

        self.br_lyr = partial(
            Dense,
            activation='linear',
            kernel_initializer=self.initializer,
            bias_initializer=self.initializer_bias,
            kernel_regularizer=self.regularizer,
            name="",
        )

        self.trk_lyr = partial(
            Dense,
            activation='linear',
            kernel_initializer=self.initializer,
            bias_initializer=self.initializer_bias
        )

        """
        Initializing network sizes and shapes for Branch network #1 (DOE input)
        """
        self.br1_l1 = self.br_lyr(self.mlp_nodes_br[0])
        self.br1_drp1 = Dropout(self.dropout_rate)
        self.br1_norm1 = BatchNormalization()
        self.br1_l2 = self.br_lyr(self.mlp_nodes_br[1])
        self.br1_drp2 = Dropout(self.dropout_rate)
        self.br1_norm2 = BatchNormalization()
        self.br1_l3 = self.br_lyr(self.mlp_nodes_br[2])
        self.br1_drp3 = Dropout(self.dropout_rate)
        self.br1_norm3 = BatchNormalization()
        self.br1_l4 = self.br_lyr(self.mlp_nodes_br[3])
        self.br1_drp4 = Dropout(self.dropout_rate)
        self.br1_norm4 = BatchNormalization()
        self.br1_l7 = self.br_lyr(
            self.latent_dim, activation='selu', name="br1_end"
        )

        """
        Initializing network sizes for trunk network
        """
        self.trk1_l1 = self.trk_lyr(self.mlp_nodes_trk[0])
        self.batchnorm_trk1_l1 = BatchNormalization()
        self.trk1_l2 = self.trk_lyr(self.mlp_nodes_trk[1])
        self.batchnorm_trk1_l2 = BatchNormalization()
        self.trk1_l3 = self.trk_lyr(self.mlp_nodes_trk[2])
        self.batchnorm_trk1_l3 = BatchNormalization()
        self.trk1_l4 = self.trk_lyr(self.mlp_nodes_trk[3])
        self.batchnorm_trk1_l4 = BatchNormalization()
        self.trk1_l5 = self.trk_lyr(self.mlp_nodes_trk[4])
        self.batchnorm_trk1_l5 = BatchNormalization()
        self.trk1_l6 = self.trk_lyr(self.mlp_nodes_trk[5])
        self.batchnorm_trk1_l6 = BatchNormalization()
        self.trk1_l7 = self.trk_lyr(self.mlp_nodes_trk[6])
        self.batchnorm_trk1_l7 = BatchNormalization()
        self.trk1_l8 = self.trk_lyr(self.mlp_nodes_trk[7])
        self.batchnorm_trk1_l8 = BatchNormalization()
        self.trk1_l9 = self.trk_lyr(self.mlp_nodes_trk[8])
        self.batchnorm_trk1_l9 = BatchNormalization()
        self.trk1_l10 = self.trk_lyr(self.latent_dim)

        """
        Projection layers
        """
        self.trk1_proj_lyr1 = self.trk_lyr(self.mlp_nodes_trk[2], name='proj_l1_trk1')
        self.trk1_proj_lyr2 = self.trk_lyr(self.mlp_nodes_trk[3], name='proj_l2_trk1')
        self.trk1_proj_lyr3 = self.trk_lyr(self.mlp_nodes_trk[4], name='proj_l3_trk1')
        self.trk1_proj_lyr4 = self.trk_lyr(self.mlp_nodes_trk[5], name='proj_l4_trk1')
        self.trk1_proj_lyr5 = self.trk_lyr(self.mlp_nodes_trk[6], name='proj_l5_trk1')
        self.trk1_proj_lyr6 = self.trk_lyr(self.mlp_nodes_trk[7], name='proj_l6_trk1')
        self.trk1_proj_lyr7 = self.trk_lyr(self.mlp_nodes_trk[8], name='proj_l7_trk1')

    """
    Initializing call function to instantiate model. training = true should be passed here from training loop
    """

    def call_deepnet_proj(self, br_input, trunk_in, training=None):
        """
        Forward pass for network with projection layers in trunk
        :param br_input: Inputs for all branches with shape batch size x features x number of branch networks
        :param trunk_in: Inputs for all trunk network with shape features x number of trunk networks
        :param training: boolean to determine whether network is training or testing
        :return: Forward pass results for all batch networks and trunk networks
        """

        """
        Forward pass for branch network #1
        """

        x_branch1 = self.br1_l1(br_input)
        x_branch1 = self.br1_norm1(x_branch1, training=training)
        x_branch1 = self.br_act(x_branch1)
        x_branch1 = self.br1_drp1(x_branch1, training=training)
        x_branch1 = self.br1_l2(x_branch1)
        x_branch1 = self.br1_norm2(x_branch1, training=training)
        x_branch1 = self.br_act(x_branch1)
        x_branch1 = self.br1_drp2(x_branch1, training=training)
        x_branch1 = self.br1_l3(x_branch1)
        x_branch1 = self.br1_norm3(x_branch1, training=training)
        x_branch1 = self.br_act(x_branch1)
        x_branch1 = self.br1_l4(x_branch1)
        x_branch1 = self.br1_norm4(x_branch1, training=training)
        x_branch1 = self.br_act(x_branch1)
        x_branch1 = self.br1_l7(x_branch1)

        """
        Forward pass for trunk net with residual connections
        """
        x_trk1_l1 = self.trk1_l1(trunk_in)
        x_trk1_l1 = self.batchnorm_trk1_l1(x_trk1_l1, training=training)
        x_trk1_l1 = self.trk_act(x_trk1_l1)

        x_trk1_l2 = self.trk1_l2(x_trk1_l1)
        x_trk1_l2 = self.batchnorm_trk1_l2(x_trk1_l2, training=training)
        x_trk1_l2 = self.trk_act(x_trk1_l2)

        x_trk1_l3 = self.trk1_l3(x_trk1_l2)
        x_trk1_l3 = self.batchnorm_trk1_l3(x_trk1_l3, training=training)
        x_trk1_l3 = self.trk_act(x_trk1_l3)
        x_trk1_l3 = self.trk1_proj_lyr1(x_trk1_l1) + x_trk1_l3

        x_trk1_l4 = self.trk1_l4(x_trk1_l3)
        x_trk1_l4 = self.batchnorm_trk1_l4(x_trk1_l4, training=training)
        x_trk1_l4 = self.trk_act(x_trk1_l4)
        x_trk1_l4 = self.trk1_proj_lyr2(x_trk1_l2) + x_trk1_l4

        x_trk1_l5 = self.trk1_l5(x_trk1_l4)
        x_trk1_l5 = self.batchnorm_trk1_l5(x_trk1_l5, training=training)
        x_trk1_l5 = self.trk_act(x_trk1_l5)
        x_trk1_l5 = self.trk1_proj_lyr3(x_trk1_l3) + x_trk1_l5

        x_trk1_l6 = self.trk1_l6(x_trk1_l5)
        x_trk1_l6 = self.batchnorm_trk1_l6(x_trk1_l6, training=training)
        x_trk1_l6 = self.trk_act(x_trk1_l6)
        x_trk1_l6 = self.trk1_proj_lyr4(x_trk1_l4) + x_trk1_l6

        x_trk1_l7 = self.trk1_l7(x_trk1_l6)
        x_trk1_l7 = self.batchnorm_trk1_l7(x_trk1_l7, training=training)
        x_trk1_l7 = self.trk_act(x_trk1_l7)
        x_trk1_l7 = self.trk1_proj_lyr5(x_trk1_l5) + x_trk1_l7

        x_trk1_l8 = self.trk1_l8(x_trk1_l7)
        x_trk1_l8 = self.batchnorm_trk1_l8(x_trk1_l8, training=training)
        x_trk1_l8 = self.trk_act(x_trk1_l8)
        x_trk1_l8 = self.trk1_proj_lyr6(x_trk1_l6) + x_trk1_l8

        x_trk1_l9 = self.trk1_l9(x_trk1_l8)
        x_trk1_l9 = self.batchnorm_trk1_l9(x_trk1_l9, training=training)
        x_trk1_l9 = self.trk_act(x_trk1_l9)
        x_trk1_l9 = self.trk1_proj_lyr7(x_trk1_l7) + x_trk1_l9

        x_trunk1 = self.trk1_l10(x_trk1_l9)

        return (
            x_branch1,
            x_trunk1
        )
