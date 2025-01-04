"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: model script for Fisher problem
Contact: varun_kumar2@brown.edu
"""

import tensorflow as tf
from model import Deep_Net_engine
from dataset import DataSet
import os
from time import perf_counter
from plots import *
from tqdm import tqdm
import json, yaml


def main():
    def error_norm(pred, target):
        L2_norm = tf.norm(pred - target, ord=2) / tf.norm(target, ord=2)
        return L2_norm

    def data_norm(pred, target, train_mean, train_std):
        pred_norm = pred * (train_std + 1e-9) + train_mean
        target_norm = target * (train_std + 1e-9) + train_mean
        return pred_norm, target_norm

    @tf.function(jit_compile=True)
    def train_step(br_input, trunk_in, target):
        """
        :param br_input: IC+PDE params (bs,1,68)
        :param trunk_in: domain coords (bs, 1280, 2)
        :return: MSE Loss
        """

        with tf.GradientTape(persistent=False) as tape:
            br1_out, trk_pred = model.call_don(br_input, trunk_in, training=True)

            # Count trainable parameters
            # trainable_params = tf.reduce_sum([tf.reduce_prod(v.get_shape()) for v in model.trainable_variables])

            br_output = br1_out
            u1_pred = tf.einsum("ijk,mlk->il", br_output, trk_pred)

            mse_loss = tf.reduce_mean(tf.square(u1_pred - target))
            batch_loss = mse_loss + sum(model.losses)

        gradient_wt = tape.gradient(
            batch_loss, model.trainable_variables
        )

        optimizer.apply_gradients(
            zip(gradient_wt, model.trainable_variables)
        )

        del tape

        return (mse_loss,
                batch_loss
                )

    @tf.function(jit_compile=True)
    def eval_step(br_input, trunk_in, target, train_mean, train_std):

        br1_out, trk_pred = model.call_don(br_input, trunk_in, training=False)
        br_output = br1_out
        u1_pred = tf.einsum("ijk,mlk->il", br_output, trk_pred)
        u1_pred_norm, target_norm = data_norm(u1_pred, target, train_mean, train_std)
        batch_error1 = error_norm(u1_pred_norm, target_norm)

        return (
            batch_error1,
            u1_pred_norm,
            target_norm
        )

    """
    Loading model and experiment setup
    """
    stream = open("network_setup_fisher.yaml", "r")
    setup_dict = yaml.safe_load(stream)

    br_nodes = setup_dict["Architecture"]["br_lyr"]
    trk_nodes = setup_dict["Architecture"]["trk_lyr"]
    dropout = setup_dict["Architecture"]["dropout_br"]
    latent_dim = setup_dict["Architecture"]["latent_dim"]
    reg_rate = setup_dict["Architecture"]["regularizer_lambda"]
    br_act = setup_dict["Architecture"]["br_activation"]
    trk_act = setup_dict["Architecture"]["trk_activation"]

    exp_name = setup_dict["Exp_setup"]["exp_name"]
    epochs = setup_dict["Exp_setup"]["epochs"]
    test_condition = setup_dict["Exp_setup"]["test_condition"]

    """
    Setting up results directories
    """
    current_directory = os.getcwd()
    case = f"multitask_mionet_{exp_name}"
    results_dir = "/" + case + "/Results"
    plots_dir = "/" + case + "/plots"
    save_results_to = current_directory + results_dir
    save_plots_to = current_directory + plots_dir

    os.makedirs(save_results_to, exist_ok=True)
    os.makedirs(save_plots_to, exist_ok=True)

    with open(current_directory + "/" + case + "/setup.json", "w") as outfile:
        json.dump(setup_dict, outfile)
    print("\n" "Saved JSON and keras model")

    tf_datatype = tf.float32
    batch_sz_train = 200
    batch_sz_test = 800

    data = DataSet(batch_sz_train, tf_datatype)
    model = Deep_Net_engine(br_nodes, br_act, reg_rate, dropout, trk_nodes, trk_act, latent_dim)
    *_, u_train_mean_all, u_train_std_all = data.minibatch()

    """
    Training setup
    """
    initial_learning_rate1 = 1e-3
    decay_steps1 = 1000
    decay_rate1 = 0.95
    staircase = True

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate1,
        decay_steps=decay_steps1,
        decay_rate=decay_rate1,
        staircase=staircase
    )
    optimizer = tf.keras.optimizers.Adam(lr_schedule1, beta_1=0.9, beta_2=0.999)
    batch_losstot_arr = []
    batch_lossmse_arr = []
    batch_err_arr = []
    epoch_err_arr = []
    epoch_lossmse_arr = []
    epoch_losstot_arr = []
    checkpoints_path = f"./checkpts/DeepONet_Fisher_{exp_name}.ckpt"

    start_time = perf_counter()

    for epoch in range(epochs):
        """
        Training and validation steps
        """
        x_train, br_train, u_train, Xmin, Xmax, u_train_mean, u_train_std = data.minibatch()
        test_id, x_test, br_test, u_test = data.testbatch(batch_sz_test)

        mse_loss, total_loss = train_step(br_train, x_train, u_train)
        batch_lossmse_arr.append(mse_loss.numpy())
        batch_losstot_arr.append(total_loss.numpy())

        l2_err, *_ = eval_step(br_test, x_test, u_test, u_train_mean, u_train_std)
        batch_err_arr.append(l2_err.numpy())

        if epoch % 20 == 0:
            time_step_100 = perf_counter()
            comp_time = time_step_100 - start_time
            print(
                "\n",
                f"Epoch {epoch}, Total mse loss --> {mse_loss.numpy()}, validation error -->{l2_err.numpy()}, Computation time --> {comp_time}")
            print("*" * 100)
            start_time = perf_counter()

    print('\n' 'Training completed')
    model.save_weights(checkpoints_path, overwrite=True)

    loss_plot = PlotLoss(save_results_to + f'/Fisher_loss_{exp_name}.png')
    loss_plot.plot_loss(batch_lossmse_arr, batch_err_arr)
    print('\n' 'Loss plot saved')

    """
    Model evaluation and saving outputs
    """
    model.load_weights(checkpoints_path)

    test_id, x_test, br_test, u_test = data.testbatch(batch_sz_test)
    l2_err, pred, target = eval_step(br_test, x_test, u_test, u_train_mean_all, u_train_std_all)
    print('\n', f'Test L2 relative error: {l2_err.numpy()}')

    np.savez_compressed(save_results_to + f'/Fisher_DeepONet_{exp_name}.npz', model_out=pred.numpy(), gt=target.numpy(),
                        l2_error=l2_err.numpy(), dom=x_test.numpy(),
                        epoch_loss=epoch_lossmse_arr, epoch_err=epoch_err_arr, batch_loss=batch_lossmse_arr,
                        batch_err=batch_err_arr)
    print('\n' 'Data saved successfully')

    """
    Plotting results
    """
    s, t = 64, 20

    x = np.linspace(0, 1, s)
    z = np.linspace(0, 1, t)

    pred_grid = np.reshape(pred, [-1, 20, 64])
    target_grid = np.reshape(target, [-1, 20, 64])
    abs_err = np.abs(pred_grid - target_grid)
    plot_data = np.stack([pred_grid, target_grid, abs_err], axis=3)

    tt, xx = np.meshgrid(z, x, indexing='ij')
    plot_cases = [19, 47, 21, 31]
    for i in tqdm(plot_cases, desc='Results plot'):
        plot_Fisher_separate(tt, xx, plot_data[i], save_plots_to + f'/Fisher_DeepONet_{exp_name}_case{i}.png')

    pass


if __name__ == "__main__":
    main()
