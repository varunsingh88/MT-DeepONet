"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: main script for heat transfer in 3D plate
Contact: varun_kumar2@brown.edu
"""

import numpy as np
import os
from model import *
from preprocessor_don import preprocess_data
from generic_funcs import loss_function as train_loss, error_norm as L2_error, regular_0to1 as data_regular, \
    normalizer_0to1 as normalizer, PlotLoss, check_dir_exists, plot_mesh
from tqdm import tqdm
import yaml, json
from time import perf_counter


def main():
    @tf.function(jit_compile=True)
    def train_step1(br_input, trunk_in, target, masking):
        """
        Training step
        :param br_input: Geometry parameters (bs,1,2)
        :param trunk_in: Domain coords (1, N*N*N, 3)
        :param target: Output labels (bs, N*N*N)
        :param masking: Geometry mask (bs, N*N*N)
        :return: training loss (.)
        """

        with tf.GradientTape(persistent=False) as tape:
            br1_pred, trk_pred = model_deepnet.call_deepnet_proj(br_input, trunk_in, training=True)

            br_output = br1_pred
            u1_pred = tf.einsum("ijk,mlk->il", br_output, trk_pred) * masking

            mse_loss = train_loss(u1_pred, target)
            batch_loss = mse_loss + sum(model_deepnet.losses)

        gradient_wt = tape.gradient(
            batch_loss, model_deepnet.trainable_variables
        )

        optimizer1.apply_gradients(
            zip(gradient_wt, model_deepnet.trainable_variables)
        )

        del tape

        return mse_loss

    @tf.function(jit_compile=True)
    def train_step2(br_input, trunk_in, target, masking):
        """
        Training step
        :param br_input: Geometry parameters (bs,1,2)
        :param trunk_in: Domain coords (1, N*N*N, 3)
        :param target: Output labels (bs, N*N*N)
        :param masking: Geometry mask (bs, N*N*N)
        :return: training loss (.)
        """

        with tf.GradientTape(persistent=False) as tape:
            br1_pred, trk_pred = model_deepnet.call_deepnet_proj(br_input, trunk_in, training=True)

            br_output = br1_pred
            u1_pred = tf.einsum("ijk,mlk->il", br_output, trk_pred) * masking

            mse_loss = train_loss(u1_pred, target)
            batch_loss = mse_loss + sum(model_deepnet.losses)

        gradient_wt = tape.gradient(
            batch_loss, model_deepnet.trainable_variables
        )

        optimizer2.apply_gradients(
            zip(gradient_wt, model_deepnet.trainable_variables)
        )

        del tape

        return mse_loss

    @tf.function(jit_compile=True)
    def eval_step(br_input, trunk_in, target, target_max, target_min, masking):
        br1_pred, trk_pred = model_deepnet.call_deepnet_proj(br_input, trunk_in, training=False)

        br_output = br1_pred
        u1_pred = tf.einsum("ijk,mlk->il", br_output, trk_pred) * masking
        u1_pred_reg, u1_target_reg = data_regular(u1_pred, target, target_max, target_min)
        #
        l2_err = L2_error(u1_pred_reg, u1_target_reg)
        mse_err = train_loss(u1_pred, target)

        return (
            l2_err,
            mse_err,
            u1_pred_reg
        )

    @tf.function
    def to_tensor(inputs):
        output = tf.convert_to_tensor(inputs)
        output = tf.cast(output, dtype=datatype)
        return output

    """
    Loading network setup details
    """
    stream = open("network_setup_don.yaml", "r")
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
    adaptive_act = setup_dict["Exp_setup"]["adaptive_act"]
    residual_conn = setup_dict["Exp_setup"]["residual_conn"]
    br_normalization = setup_dict["Exp_setup"]["br_norm"]
    test_index = setup_dict["Exp_setup"]["test_case"]
    test_condition = setup_dict["Exp_setup"]["test_condition"]

    """
    Creating data storage directories
    """
    current_directory = os.getcwd()
    case = f"./don_3D_heatplate_{exp_name}"
    results_dir = "./" + case + "/Results"
    plots_dir = "./" + case + "/plots"
    checkpts = "./" + case + "/checkpts"
    save_results_to = results_dir
    save_plots_to = plots_dir
    save_checkpts_to = checkpts

    os.makedirs(save_results_to, mode=0o755, exist_ok=True)
    os.makedirs(save_plots_to, mode=0o755, exist_ok=True)
    os.makedirs(save_checkpts_to, mode=0o755, exist_ok=True)

    """
    Preprocess data for feeding into NN
    """
    datatype = tf.float32
    data_file = './Data/interpol_results.mat'

    train_index = np.arange(0, 64)
    train_index = np.delete(train_index, test_index)
    train_index = np.array(sorted(train_index))

    (
        br1_inp_train,
        br1_inp_test,
        u_train,
        u_test,
        datalen_train,
        datalen_test,
        trk_in_train,
        num_nodes_train,
        num_nodes_test,
        nan_mask_train,
        nan_mask_test
    ) = preprocess_data(data_file, train_index, test_index)

    masked_fltr_train = to_tensor(np.where(nan_mask_train, 0, 1))
    masked_fltr_test = to_tensor(np.where(nan_mask_test, 0, 1))
    ###
    # test_index = train_index  # Just for testing purpose, remove this!!!!!!!!!!!=
    pts_filtrd_every = 1

    br1_inp_train = to_tensor(br1_inp_train)
    br1_inp_test = to_tensor(br1_inp_test)
    u_train = to_tensor(u_train[:, ::pts_filtrd_every])
    u_test = to_tensor(u_test)
    trk_inp = to_tensor(trk_in_train[::pts_filtrd_every, :])
    trk_inp = tf.reshape(trk_inp, [1, -1, trk_inp.shape[1]])

    """"
    Data normalization routine
    """
    (
        br1_inp_train,
        br1_inp_test,
        br1_train_max,
        br1_train_min,
    ) = normalizer(br1_inp_train, br1_inp_test, axes=0)

    norm_u_train, norm_u_test, u_train_max, u_train_min = normalizer(
        u_train, u_test, axes=(0, 1)
    )

    train_batch_size = int(1.0 * datalen_train)
    test_batch_size = datalen_test

    """
    Creating training loop, instantiating sequential design model
    """
    model_deepnet = Deep_Net_engine(
        mlp_nodes_br=br_nodes,
        act_br=br_act,
        reg_br_lambda=reg_rate,
        drpout_rate_br=dropout,
        mlp_nodes_trk=trk_nodes,
        act_trk=trk_act,
        latent_dim=latent_dim,
        adaptive_act=adaptive_act,
        residual_conn=residual_conn
    )

    """
    Training setup
    """
    initial_learning_rate1 = 1e-3
    decay_steps1 = 1000
    decay_rate1 = 0.9
    staircase = True

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate1,
        decay_steps=decay_steps1,
        decay_rate=decay_rate1,
        staircase=staircase
    )

    optimizer1 = tf.keras.optimizers.Adam(lr_schedule1, beta_1=0.9, beta_2=0.95)

    initial_learning_rate2 = 1e-5
    decay_steps2 = 100
    decay_rate2 = 0.9
    staircase = True

    lr_schedule2 = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate2,
        decay_steps=decay_steps2,
        decay_rate=decay_rate2,
        staircase=staircase
    )

    optimizer2 = tf.keras.optimizers.Lion(lr_schedule2, beta_1=0.9, beta_2=0.99)

    loss_array = []
    error_l2_arr = []
    error_mse_arr = []

    chckpt_path_end = f'./checkpts/'
    check_dir_exists(chckpt_path_end)

    checkpoints_path = chckpt_path_end + f'DON_3D_plate_{exp_name}.ckpt'

    print(
        f"Starting training for DeepONet for 3D plate heat transfer {exp_name}"
    )
    start_time = perf_counter()

    for epoch in tqdm(range(epochs), desc='Network training'):
        shfl_indx = tf.random.shuffle(tf.range(0, train_batch_size))
        br1_inp_shfld = tf.gather(br1_inp_train, shfl_indx)[
                        0:train_batch_size]
        target_train_shfld = tf.gather(norm_u_train, shfl_indx)[0:train_batch_size]
        mask_train_shfld = tf.gather(masked_fltr_train, shfl_indx)[0:train_batch_size]

        """
        Training step
        """
        if epoch <= 15000:
            loss = train_step1(
                br1_inp_shfld,
                trk_inp,
                target_train_shfld,
                mask_train_shfld
            )
            loss_array.append(float(loss.numpy()))
        else:
            loss = train_step2(
                br1_inp_shfld,
                trk_inp,
                target_train_shfld,
                mask_train_shfld
            )
            loss_array.append(float(loss.numpy()))

        """
        Testing step
        """
        l2_err, mse_err, _ = eval_step(
            br1_inp_test,
            trk_inp,
            norm_u_test,
            u_train_max,
            u_train_min,
            masked_fltr_test
        )

        error_l2_arr.append(float(l2_err.numpy()))
        error_mse_arr.append(float(mse_err.numpy()))

        if epoch % 100 == 0:
            time_step_100 = perf_counter()
            cons_time = time_step_100 - start_time
            print(
                "\n",
                f'Epoch -->{epoch}, mse loss --> {loss.numpy()}, L2 relative error --> {l2_err.numpy()}, comp time --> {cons_time}'
            )

            start_time = perf_counter()

    model_deepnet.save_weights(checkpoints_path, overwrite=True)

    print("\n" "Training complete")

    """
    Plotting loss
    """
    loss_plot = PlotLoss(save_plots_to + f'/DON_3D_plate_{exp_name}.png')
    loss_plot.plot_loss(loss_array, error_mse_arr)
    print('\n' 'Loss plot saved')

    with open(current_directory + "/" + case + "/setup.json", "w") as outfile:
        json.dump(setup_dict, outfile)

    print("\n" "Saved JSON and keras model")

    """
    Loading saved weights and creating Predictions section
    """
    model_deepnet.load_weights(checkpoints_path)
    # #
    l2_err, mse_err, preds = eval_step(
        br1_inp_test,
        trk_inp,
        norm_u_test,
        u_train_max,
        u_train_min,
        masked_fltr_test
    )
    print('\n' f'L2 error: {l2_err.numpy()}')
    np.savetxt(save_results_to + f'/L2_rel_err_{exp_name}.txt', np.array(l2_err, ndmin=1))

    np.savez_compressed(save_results_to + f'/DON_3D_plate_{exp_name}_nan_mask.npz', u_pred=preds.numpy(),
                        u_target=u_test.numpy(), loss=loss_array,
                        error_l2=error_l2_arr,
                        error_mse=error_mse_arr,
                        coords=trk_inp.numpy(),
                        test_indx=test_index,
                        train_index=train_index,
                        nan_mask_test=nan_mask_test
                        )

    """
    Plotting visualization
    """
    dom_pts = tf.tile(trk_inp, [test_batch_size, 1, 1])
    for i in tqdm(range(5, 30), desc='Creating prediction plots'):
        mesh_pts = dom_pts[i:i + 1, :, :][~nan_mask_test[i:i + 1, :]]
        target_geom = u_test[i:i + 1, :][~nan_mask_test[i:i + 1, :]]
        pred_geom = preds[i:i + 1, :][~nan_mask_test[i:i + 1, :]]
        filename = save_plots_to + f'/temp_profile_sample{test_index[i]}_{exp_name}_relerr.png'
        plot_mesh(mesh_pts.numpy(), pred_geom.numpy(), target_geom.numpy(), filename)

    print('\n' 'All plots saved successfully')
    pass


if __name__ == "__main__":
    main()
