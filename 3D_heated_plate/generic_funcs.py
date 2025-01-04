"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: generic functions for heat transfer in 3D plate
Contact: varun_kumar2@brown.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
# import pyvista as pv


def loss_function(prediction, target):
    loss = tf.math.reduce_mean(tf.square((prediction - target)))
    return loss


def error_norm(prediction, target):
    L2_error = tf.norm(prediction - target, ord='fro', axis=(0, 1)) / tf.norm(target, ord='fro', axis=(0, 1))
    return L2_error


def normalizer_0to1(train_data, test_data, axes=0):
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


def regular_0to1(u_pred_norm, u_target_norm, train_max, train_min):
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


def data_normalizer(train_data, test_data):
    """
    :param train_data:
    :param test_data:
    :return: Normalized data  [rows x columns]

    Function performs min-max scaling between -1 and 1
    """
    max_col = tf.math.reduce_max(train_data, axis=(0, 1))
    min_col = tf.math.reduce_min(train_data, axis=(0, 1))
    train_inputs_norm = 2 * (train_data - min_col) / (max_col - min_col) - 1
    test_data_norm = 2 * (test_data - min_col) / (max_col - min_col) - 1

    return train_inputs_norm, test_data_norm, max_col, min_col


def data_standardizer(train_data, test_data):
    """
    :param train_data:
    :param test_data:
    :return: Normalized data wrt mean and standard deviation

    Function performs mean-standard dev scaling
    """
    mean_col = tf.math.reduce_mean(train_data, axis=0)
    stdev_col = tf.math.reduce_std(train_data, axis=0)
    train_inputs_norm = (train_data - mean_col) / stdev_col
    test_data_norm = (test_data - mean_col) / stdev_col

    return train_inputs_norm, test_data_norm, mean_col, stdev_col


def data_regular(u_pred_norm, u_target_norm, train_max, train_min):
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


# @tf.function()
def data_regular_std(pred_norm, target_norm, train_mean, train_stddev):
    """
    :param pred_norm:
    :param target_norm:
    :param train_mean:
    :param train_stddev:
    :return:

    Function converts data scaled using mean and stddev back to regular scale
    """
    pred_reg = pred_norm * train_stddev + train_mean
    target_reg = target_norm * train_stddev + train_mean
    return pred_reg, target_reg


"""
Function to check if specified directory exists or not
"""


def check_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


"""
Class containing individual sub-plot methods for loss
"""


class PlotLoss:

    def __init__(self, filename):
        self.filename = filename

    def plot_loss(self, dist_loss, dist_error):
        fig = plt.figure(figsize=(21, 7))
        ax1 = fig.add_subplot(121)
        line1 = ax1.plot(dist_loss, color="red", label='mse_loss')
        ax1.set_xlabel("epochs")
        ax1.set_yscale("log")
        plt.legend(loc="upper right")

        ax2 = fig.add_subplot(122)
        line2 = ax2.plot(dist_error, color="blue", label='L2 rel error')
        ax2.set_xlabel("epochs")
        ax2.set_yscale("log")
        plt.legend(loc="upper right")
        # plt_axis.set_title(title_name)
        plt.tight_layout()
        plt.savefig(self.filename)
        # plt.show()

    def set_main_title(self, title):
        self.fig.suptitle(title)

    def show_plot(self, row, col, loss, loss_label):
        """
        Special function to either clean the last plot when not required or to plot different behavior compared to regular subplot routine above
        :param row:
        :param col:
        :param loss:
        :param loss_label:
        :return:
        """
        plt_axis = self.axes[row, col]
        line1 = plt_axis.plot(loss, color="red", label=loss_label)
        plt_axis.set_yscale("log")
        plt_axis.legend(loc="upper right")
        # plt_axis.set_axis_off()

        plt.tight_layout()
        plt.savefig(self.filename)
        # plt.show()
        print("\n" "Loss plot saved successfully")


"""
Plotting mesh type layout with PyVista
"""


def plot_mesh(mesh_pts, pred, target, filename):
    """
    Function to plot results for comparison. Enable pyvista module import to use

    :param mesh_pts: 3D point cloud num_pts x 3
    :param pred: DeepONet predictions num_pts x 1
    :param target: Ground truth num_pts x 1
    :param filename: Filename to save the plot
    :return:
    """

    sargs1 = dict(
        height=0.1,
        vertical=False,
        position_x=0.2,
        position_y=0.7,
        title_font_size=14,
        label_font_size=12,
        title=" "
    )

    sargs2 = dict(
        height=0.1,
        vertical=False,
        position_x=0.2,
        position_y=0.7,
        title_font_size=14,
        label_font_size=12,
        title="  "
    )

    sargs3 = dict(
        height=0.1,
        vertical=False,
        position_x=0.2,
        position_y=0.7,
        title_font_size=14,
        label_font_size=12,
        title="   "
    )

    abs_err = np.abs(target - pred)
    p = pv.Plotter(notebook=False, shape=(1, 3))
    # p.add_mesh(grid, scalars=grid_z0, cmap="jet", render_points_as_spheres=True, point_size=9.0)
    p.subplot(0, 0)
    p.add_mesh(mesh_pts[::2, :], scalars=target[::2], cmap="plasma", render_points_as_spheres=True, point_size=9.0,
               scalar_bar_args=sargs1)
    p.add_text('Target', position='lower_edge', font_size=10, color='blue', font='arial', shadow=True)
    p.subplot(0, 1)
    p.add_mesh(mesh_pts[::2, :], scalars=pred[::2], cmap="plasma", render_points_as_spheres=True, point_size=9.0,
               scalar_bar_args=sargs2, clim=[np.min(target), np.max(target)])  # , clim=[np.min(target), np.max(target)]
    p.add_text('Prediction', position='lower_edge', font_size=10, color='blue', font='arial', shadow=True)
    p.subplot(0, 2)
    p.add_mesh(mesh_pts[::2, :], scalars=abs_err[::2], cmap="plasma", render_points_as_spheres=True, point_size=9.0,
               scalar_bar_args=sargs3)
    p.add_text('Absolute error', position='lower_edge', font_size=10, color='blue', font='arial', shadow=True)
    p.link_views()
    p.save_graphic(filename + ".svg")
    # p.show()
    p.close()
