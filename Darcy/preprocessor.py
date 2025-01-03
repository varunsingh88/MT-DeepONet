"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: main script for training source model for Darcy flow problem
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import random
import sys

np.random.seed(1234)


class DataSet:
    def __init__(self, bs):
        # self.num_training = num_training
        self.bs = bs

    def to_tensor(self, inputs, datatype):
        output = tf.convert_to_tensor(inputs)
        output = tf.cast(output, dtype=datatype)
        return output

    def normalizer_0to1(self, train_data, test_data, axes=(0)):
        """
        :param train_data:
        :param test_data:
        :return: Normalized data  [rows x columns]

        Function performs min-max scaling between 0 and 1
        """
        max_col = tf.math.reduce_max(train_data, axis=axes)
        min_col = tf.math.reduce_min(train_data, axis=axes)
        train_inputs_norm = (train_data - min_col) / (max_col - min_col)
        test_data_norm = (test_data - min_col) / (max_col - min_col)

        return train_inputs_norm, test_data_norm, max_col, min_col

    def regular_0to1(self, u_pred_norm, u_target_norm, train_max, train_min):
        """

        :param u_pred_norm:
        :param u_target_norm:
        :param train_max: max value of training data from normalizer function
        :param train_min: min value of training data from normalizer function
        :return:

        Function converts data scaled using min-max values to regular scale from 0 to 1
        """
        u_pred_regular = (u_pred_norm) * (train_max - train_min) + train_min
        u_target_regular = (u_target_norm) * (train_max - train_min) + train_min
        return u_pred_regular, u_target_regular

    def data_normalizer(self, train_data, axes=(0), keep_dims=None):
        """
        :param train_data:
        :param test_data:
        :return: Normalized data  [rows x columns]

        Function performs min-max scaling between -1 and 1
        """
        max_col = tf.math.reduce_max(train_data, axis=axes, keepdims=keep_dims)
        min_col = tf.math.reduce_min(train_data, axis=axes, keepdims=keep_dims)
        train_inputs_norm = 2 * (train_data - min_col) / (max_col - min_col) - 1
        # test_data_norm = 2 * (test_data - min_col) / (max_col - min_col) - 1

        return train_inputs_norm

    def data_regular(self, u_pred_norm, u_target_norm, train_max, train_min):
        """

        :param u_pred_norm:
        :param u_target_norm:
        :param train_max: max value of training data from normalizer function
        :param train_min: min value of training data from normalizer function
        :return:

        Function converts data scaled using min-max values to regular scale from -1 to 1
        """
        u_pred_regular = (u_pred_norm + 1) * (train_max - train_min) / 2 + train_min
        u_target_regular = (u_target_norm + 1) * (train_max - train_min) / 2 + train_min
        return u_pred_regular, u_target_regular

    def reg_to_gaussian(self, train_data, test_data, tol_factor, axes):
        """
        :param train_data:
        :param test_data:
        :return: Normalized data wrt mean and standard deviation

        Function performs mean-standard dev scaling
        """
        mean_col = tf.math.reduce_mean(train_data, axis=axes, keepdims=True)
        stdev_col = tf.math.reduce_std(train_data, axis=axes, keepdims=True)
        train_inputs_norm = (train_data - mean_col) / (stdev_col + tol_factor)
        test_data_norm = (test_data - mean_col) / (stdev_col + tol_factor)

        return train_inputs_norm, test_data_norm, mean_col, stdev_col

    def gaussian_to_reg(self, pred_norm, target_norm, train_mean, train_stddev, tol_factor):
        """
        :param pred_norm:
        :param target_norm:
        :param train_mean:
        :param train_stddev:
        :return:

        Function converts data scaled using mean and stddev back to regular scale
        """
        pred_reg = pred_norm * (train_stddev + tol_factor) + train_mean
        target_reg = target_norm * (train_stddev + tol_factor) + train_mean
        return pred_reg, target_reg

    def load_data_source(self, num_training, num_geom, tf_datatype):
        """
        :param num_training: Num of training samples for each geometry (int)
        :param num_geom: Number of geometries
        :param tf_datatype: data precision (default: tf.float32)
        :return: F_train (bsx100x100x3),
                U_train (bsx10000x1),
                F_test (bsx100x100x3),
                U_test (bsx10000x1),
                mask_train (bsx10000x1),
                mask_test (bsx10000x1),
                dom (1x10000x2)
        """

        file1 = loadmat('./Data/Dataset_sq_mask.mat')
        file2 = loadmat('./Data/Dataset_circle_mask.mat')
        file3 = loadmat('./Data/Dataset_eqtri_mask.mat')

        k1_train = file1['k_train']
        np.random.seed(100)
        train_index = random.sample(range(0, k1_train.shape[0]), num_training)
        train_index = np.array(train_index)

        k1_train = file1['k_train'][train_index]
        u1_train = file1['u_train'][train_index]
        f1_train = file1['shape_train'][train_index]

        k1_test = file1['k_test']
        u1_test = file1['u_test']
        f1_test = file1['shape_test']

        if num_geom > 1:
            k2_train = file2['k_train'][train_index]
            u2_train = file2['u_train'][train_index]
            f2_train = file2['shape_train'][train_index]

            k2_test = file2['k_test']
            u2_test = file2['u_test']
            f2_test = file2['shape_test']

            k3_train = file3['k_train'][train_index]
            u3_train = file3['u_train'][train_index]
            f3_train = file3['shape_train'][train_index]
            #
            k3_test = file3['k_test']
            u3_test = file3['u_test']
            f3_test = file3['shape_test']

            """
            Concatenating conductivity fields
            """
            k_train = np.concatenate((k1_train, k2_train, k3_train), axis=0)
            k_test = np.concatenate((k1_test, k2_test, k3_test), axis=0)

            f_train = np.concatenate((f1_train, f2_train, f3_train), axis=0)
            f_test = np.concatenate((f1_test, f2_test, f3_test), axis=0)

            u_train = np.concatenate((u1_train, u2_train, u3_train), axis=0)
            u_test = np.concatenate((u1_test, u2_test, u3_test), axis=0)

        """
        Use the following 6 lines when dealing with single geometry
        """
        if num_geom == 1:
            k_train = k1_train
            k_test = k1_test
            f_train = f1_train
            f_test = f1_test
            u_train = u1_train
            u_test = u1_test

        k_train = np.log(k_train)
        k_test = np.log(k_test)

        s = 100
        r = s * s

        xx = file1['xx']
        yy = file1['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        k_train_mean = np.mean(np.reshape(k_train, (-1, s, s)), 0)
        k_train_std = np.std(np.reshape(k_train, (-1, s, s)), 0)
        k_train_mean = np.reshape(k_train_mean, (-1, s, s, 1))
        k_train_std = np.reshape(k_train_std, (-1, s, s, 1))
        k_train = np.reshape(k_train, (-1, s, s, 1))
        k_train = (k_train - k_train_mean) / (k_train_std)
        k_test = np.reshape(k_test, (-1, s, s, 1))
        k_test = (k_test - k_train_mean) / (k_train_std)

        f_train = np.reshape(f_train, (-1, s, s, 1))
        f_test = np.reshape(f_test, (-1, s, s, 1))

        F_train = np.concatenate((k_train, f_train), axis=-1)
        F_test = np.concatenate((k_test, f_test), axis=-1)

        U_train = np.reshape(u_train, (-1, r)) * 10
        U_test = np.reshape(u_test, (-1, r)) * 10

        mask_train = np.reshape(f_train, (-1, s * s), order="F")
        mask_test = np.reshape(f_test, (-1, s * s), order="F")

        F_train = self.to_tensor(F_train, tf_datatype)
        dom = self.to_tensor(X, tf_datatype)
        U_train = self.to_tensor(U_train, tf_datatype)
        F_test = self.to_tensor(F_test, tf_datatype)
        U_test = self.to_tensor(U_test, tf_datatype)
        mask_train = self.to_tensor(mask_train, tf_datatype)
        mask_test = self.to_tensor(mask_test, tf_datatype)

        dom = tf.reshape(dom, [1, dom.shape[0], dom.shape[1]])
        dom = self.data_normalizer(dom, axes=1, keep_dims=True)  # Normalizing domain from -1 to 1

        return F_train, U_train, F_test, U_test, mask_train, mask_test, dom

    def load_data_target(self, num_training, num_testing, tf_datatype):
        """
        Data processor for Transfer Learning Target model
        :param num_training: Number of training samples (int)
        :param num_testing: Number of testing samples (int)
        :param tf_datatype: data precision (default float32)
        :return:
        """
        file1 = loadmat('./Data/Dataset_sq_cross_mask.mat')

        f_train_sq = file1['k_train']

        random.seed(100)
        indx = np.arange(0, f_train_sq.shape[0])
        train_index = random.sample(range(0, f_train_sq.shape[0]), num_training)
        test_index = np.delete(indx, train_index)[0:num_testing]
        train_index = np.array(train_index)

        k_train = file1['k_train'][train_index]
        u_train = file1['u_train'][train_index]
        f_train = file1['shape_train'][train_index]

        # Using slice of training data for testing
        k_test = file1['k_train'][test_index]
        u_test = file1['u_train'][test_index]
        f_test = file1['shape_train'][test_index]

        k_train = np.log(k_train)
        k_test = np.log(k_test)

        s = 100
        r = s * s

        xx = file1['xx']
        yy = file1['yy']
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))
        X = np.hstack((xx, yy))

        k_train_mean = np.mean(np.reshape(k_train, (-1, s, s)), 0)
        k_train_std = np.std(np.reshape(k_train, (-1, s, s)), 0)
        k_train_mean = np.reshape(k_train_mean, (-1, s, s, 1))
        k_train_std = np.reshape(k_train_std, (-1, s, s, 1))
        k_train = np.reshape(k_train, (-1, s, s, 1))
        k_train = (k_train - k_train_mean) / (k_train_std)
        k_test = np.reshape(k_test, (-1, s, s, 1))
        k_test = (k_test - k_train_mean) / (k_train_std)

        f_train = np.reshape(f_train, (-1, s, s, 1))
        f_test = np.reshape(f_test, (-1, s, s, 1))

        F_train = np.concatenate((k_train, f_train), axis=-1)
        F_test = np.concatenate((k_test, f_test), axis=-1)
        U_train = np.reshape(u_train, (-1, r)) * 10
        U_test = np.reshape(u_test, (-1, r)) * 10

        mask_train = np.reshape(f_train, (-1, s * s), order="F")
        mask_test = np.reshape(f_test, (-1, s * s), order="F")

        F_train = self.to_tensor(F_train, tf_datatype)
        dom = self.to_tensor(X, tf_datatype)
        U_train = self.to_tensor(U_train, tf_datatype)
        F_test = self.to_tensor(F_test, tf_datatype)
        U_test = self.to_tensor(U_test, tf_datatype)
        mask_train = self.to_tensor(mask_train, tf_datatype)
        mask_test = self.to_tensor(mask_test, tf_datatype)

        dom = tf.reshape(dom, [1, dom.shape[0], dom.shape[1]])
        dom = self.data_normalizer(dom, axes=1, keep_dims=True)  # Normalizing domain from -1 to 1

        return F_train, U_train, F_test, U_test, mask_train, mask_test, dom


def regular_scale(pred, target, mean, stddev):
    pred = pred * (stddev + 1.0e-9) + mean
    target = target * (stddev + 1.0e-9) + mean

    return pred, target


def batch_shflr(batch_sz, num_samples, br_input, mask, target):
    br_inpt_shfld = tf.random.shuffle(br_input, seed=1)[0:batch_sz]
    mask_shfld = tf.random.shuffle(mask, seed=1)[0:batch_sz]
    target_shfld = tf.random.shuffle(target, seed=1)[0:batch_sz]

    return br_inpt_shfld, mask_shfld, target_shfld
