"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: model script for Fisher problem
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
import numpy as np

# np.random.seed(1234)      # Enable for repeatable test-train data split
from sklearn.model_selection import train_test_split


class DataSet:
    def __init__(self, bs, data_type):
        self.bs = bs
        self.data_type = data_type
        self.F_train, self.U_train, self.F_test, self.U_test, \
            self.X, self.u_mean, self.u_std = self.load_data()

    def to_tensor(self, inputs):
        output = tf.convert_to_tensor(inputs)
        output = tf.cast(output, dtype=self.data_type)
        return output

    def decoder(self, x):
        x = x * (self.u_std + 1.0e-9) + self.u_mean

        return x

    def load_data(self):
        file = np.load('./Data/Fisher1_nsamples_1000_amin_0.1_amax_1.0_anum_10.npz')

        nt, nx = 20, 64
        n_samples = file['n_samples']
        inputs1 = file['inputs'][:, 1:].reshape(n_samples, nx, 1)
        fac = file['inputs'][:, 0:1].reshape(n_samples, 1, 1)
        eq_para = fac * np.tile(np.array([[1, -1, 0, 0]]), (n_samples, 1)).reshape(n_samples, 4, 1)
        inputs1 = np.concatenate((eq_para, inputs1), axis=1)
        outputs1 = np.array((file['outputs'])).reshape(n_samples, nt, nx)

        file = np.load('./Data/Fisher2_nsamples_1000_amin_0.1_amax_1.0_anum_10.npz')

        nt, nx = 20, 64
        n_samples = file['n_samples']
        inputs2 = file['inputs'][:, 1:].reshape(n_samples, nx, 1)
        fac = file['inputs'][:, 0:1].reshape(n_samples, 1, 1)
        eq_para = fac * np.tile(np.array([[1, 0, -1, 0]]), (n_samples, 1)).reshape(n_samples, 4, 1)
        inputs2 = np.concatenate((eq_para, inputs2), axis=1)
        outputs2 = np.array((file['outputs'])).reshape(n_samples, nt, nx)

        file = np.load('./Data/Fisher4_nsamples_2000_amin_0.1_amax_1.0_anum_10_bmin_0.5_bmax_2.0_bnum_10.npz')

        nt, nx = 20, 64
        n_samples = file['n_samples']
        inputs4 = file['inputs'][:, 2:].reshape(n_samples, nx, 1)
        fac = file['inputs'][:, 0:1].reshape(n_samples, 1, 1)
        beta = file['inputs'][:, 1:2]
        eq_para = fac * np.reshape(
            np.concatenate((np.exp(-beta), np.exp(-beta) * (beta - 1), np.exp(-beta) * beta * (0.5 * beta - 1), \
                            -0.5 * np.exp(-beta) * beta ** 2), axis=-1), (-1, 4, 1))
        inputs4 = np.concatenate((eq_para, inputs4), axis=1)
        outputs4 = np.array((file['outputs'])).reshape(n_samples, nt, nx)

        inputs = np.concatenate((inputs1, inputs2, inputs4), axis=0)
        outputs = np.concatenate((outputs1, outputs2, outputs4), axis=0)

        f_train, f_test, u_train, u_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

        s, t = 64, 20

        x = np.linspace(0, 1, s)
        z = np.linspace(0, 1, t)

        zz, xx = np.meshgrid(z, x, indexing='ij')
        xx = np.reshape(xx, (-1, 1))  # flatten
        zz = np.reshape(zz, (-1, 1))  # flatten

        X = np.hstack((zz, xx))  # shape=[t*s*s,3]
        coeff_train = np.reshape(f_train[:, 0:4], (-1, 1, 4))
        coeff_test = np.reshape(f_test[:, 0:4], (-1, 1, 4))
        f_train = f_train[:, 4:]
        f_test = f_test[:, 4:]

        # compute mean values
        f_train_mean = np.mean(f_train, 0)
        f_train_std = np.std(f_train, 0)
        u_train_mean = np.mean(u_train, 0)
        u_train_std = np.std(u_train, 0)

        num_res = t * s  # total output dimension

        # Reshape
        f_train_mean = np.reshape(f_train_mean, (-1, 1, s))
        f_train_std = np.reshape(f_train_std, (-1, 1, s))
        u_train_mean = np.reshape(u_train_mean, (-1, num_res))
        u_train_std = np.reshape(u_train_std, (-1, num_res))

        #  Mean normalization of train data
        F_train = np.reshape(f_train, (-1, 1, s))
        F_train = (F_train - f_train_mean) / (f_train_std + 1.0e-9)
        F_train = np.concatenate((coeff_train, F_train), axis=-1)
        U_train = np.reshape(u_train, (-1, num_res))
        U_train = (U_train - u_train_mean) / (u_train_std + 1.0e-9)

        #  Mean normalization of test data (using the mean and std of train)
        F_test = np.reshape(f_test, (-1, 1, s))
        F_test = (F_test - f_train_mean) / (f_train_std + 1.0e-9)
        F_test = np.concatenate((coeff_test, F_test), axis=-1)
        U_test = np.reshape(u_test, (-1, num_res))
        U_test = (U_test - u_train_mean) / (u_train_std + 1.0e-9)

        return F_train, U_train, F_test, U_test, X, u_train_mean, u_train_std

    def minibatch(self):
        # choose random indices - replace=False to avoid sampling same data
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)

        f_train = self.F_train[batch_id]
        u_train = self.U_train[batch_id]

        x_train = self.X

        f_train = self.to_tensor(f_train)
        u_train = self.to_tensor(u_train)
        x_train = self.to_tensor(x_train)
        u_train_mean = self.to_tensor(self.u_mean)
        u_train_std = self.to_tensor(self.u_std)
        x_train = tf.reshape(x_train, [-1, x_train.shape[0], x_train.shape[1]])

        Xmin = np.array([0., 0.]).reshape((-1, 2))
        Xmax = np.array([1., 1.]).reshape((-1, 2))

        return x_train, f_train, u_train, Xmin, Xmax, u_train_mean, u_train_std

    def testbatch(self, num_test):
        batch_id = np.arange(num_test)
        f_test = self.F_test[batch_id]
        u_test = self.U_test[batch_id]

        x_test = self.X
        f_test = self.to_tensor(f_test)
        u_test = self.to_tensor(u_test)
        x_test = self.to_tensor(x_test)
        x_test = tf.reshape(x_test, [-1, x_test.shape[0], x_test.shape[1]])

        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, x_test, f_test, u_test
