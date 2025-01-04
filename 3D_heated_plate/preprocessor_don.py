"""
Objective: To create a routine for pre-processing DeepONet data

Revision history:
09/15/2023: Rev1 --> 1st draft for baseline DeepONet

"""

import csv
import numpy as np
from scipy.io import loadmat


def preprocess_data(data_file, train_index, test_index):
    """

    :param data_file: .npz file containing mesh points and output data for all samples
    :param train_index: Indexes to be used for training
    :param test_index: Indexes used for validation and testing
    :return:
    """

    """
    Branch data creation from DOE file
    """
    exp_doe = './Data/doe.csv'
    with open(exp_doe, "r", newline="") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skipping header row
        # Convert the data to a list of lists
        data_list = [row[1:] for row in csv_reader]

    # Convert the list of lists to a NumPy array
    doe_array = np.array(data_list).astype(float)

    f1_train = doe_array[train_index, 0:2]
    data_len_train, br_arr_wd = f1_train.shape[0], f1_train.shape[1]
    num_nodes_train = doe_array[train_index, 2]

    br_in_train = np.reshape(f1_train, (data_len_train, 1, br_arr_wd))

    f1_test = doe_array[test_index, 0:2]
    data_len_test = f1_test.shape[0]
    num_nodes_test = doe_array[test_index, 2]

    br_in_test = np.reshape(f1_test, (data_len_test, 1, br_arr_wd))

    sim_data = loadmat(data_file)

    u1 = sim_data["interp_results"]  # temperature values bs, xx*yy*zz
    nan_mask = np.isnan(u1)  # Identifying location of Nan's in output due to interpolation on regular grid
    u1[nan_mask] = 0.0  # Setting all NaN's in data to 0
    u1_train = u1[train_index, :]
    u1_test = u1[test_index, :]

    u_training = u1_train
    u_testing = u1_test

    grid_data = np.concatenate([sim_data["xflat"], sim_data["yflat"], sim_data["zflat"]],
                               axis=1)  # Flattened xx, yy, zz in interpolation grid

    trk_in_train = grid_data

    return (
        br_in_train,
        br_in_test,
        u_training,
        u_testing,
        data_len_train,
        data_len_test,
        trk_in_train,
        # trk_in_test,
        num_nodes_train,
        num_nodes_test,
        nan_mask[train_index, :],
        nan_mask[test_index, :]
    )
