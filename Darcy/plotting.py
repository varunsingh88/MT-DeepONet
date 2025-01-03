"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: main script for training source model for Darcy flow problem
Contact: varun_kumar2@brown.edu

"""
import matplotlib.pyplot as plt
import numpy as np
# import pyvista as pv      # Enable this to plot Darcy results
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_darcy_mod(xx, yy, zz, pred, target, filename, save_plot=False):
    """
    Function to plot results for Darcy equation. Uses a dummy square grid overlaid to create uniform grid images
    :param xx: mesh grid x values 100 x 100
    :param yy: meshgrid y values 100 x 100
    :param zz: meshgrid z values. All 0s in this case since it is 2D plot 100 x 100
    :param pred: DeepONet prediction [10000,]
    :param target:Reference solution [10000,]
    :param filename: File name to save plots
    :return: None
    """

    sargs1 = dict(
        height=0.1,
        width=0.7,
        vertical=False,
        position_x=0.2,
        position_y=0.75,
        title_font_size=20,
        label_font_size=20,
        fmt="%.3f",
        n_labels=5,
        title="Reference"
    )

    sargs2 = dict(
        height=0.1,
        width=0.7,
        vertical=False,
        position_x=0.2,
        position_y=0.75,
        title_font_size=20,
        label_font_size=20,
        fmt="%.3f",
        n_labels=5,
        title="Prediction"
    )

    sargs3 = dict(
        height=0.1,
        width=0.7,
        vertical=False,
        position_x=0.2,
        position_y=0.75,
        title_font_size=20,
        label_font_size=20,
        fmt="%.3f",
        n_labels=5,
        title="Absolute error"
    )
    camera_pos = (10, 10, 10)
    focal_point = (0, 0, 0)

    abs_err = np.abs(target - pred)

    grid1 = pv.StructuredGrid(xx, yy, zz)
    grid1.point_data["target"] = target.flatten(order="C")
    # grid1.point_data_to_cell_data(pass_point_data=True)
    grid2 = pv.StructuredGrid(xx, yy, zz)
    grid2.point_data["target"] = target.flatten(order="C")
    grid2.point_data["pred"] = pred.flatten(order="C")
    # grid2.point_data_to_cell_data(pass_point_data=True)
    grid3 = pv.StructuredGrid(xx, yy, zz)
    grid3.point_data["target"] = target.flatten(order="C")
    grid3.point_data["error"] = abs_err.flatten(order="C")
    # grid3.point_data_to_cell_data(pass_point_data=True)

    # Threshold the scalar values to remove those that are 0 or above a desired threshold
    thresholded1 = grid1.threshold(1e-4, invert=False)
    thresholded2 = grid2.threshold(1e-4, invert=False, scalars="target")
    thresholded3 = grid3.threshold(1e-4, invert=False, scalars="target")

    p = pv.Plotter(notebook=False, shape=(1, 3), window_size=(1800, 1400), border=False)
    # p.add_mesh(grid, scalars=grid_z0, cmap="inferno", render_points_as_spheres=True, point_size=9.0)
    p.subplot(0, 0)
    p.show_grid(
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=False,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=False,
        bold=True,
        font_size=15,
        font_family="courier",
        color=None,
        xtitle="X",
        ytitle="Y",
        n_xlabels=3,
        n_ylabels=5,
        use_2d=True,
        grid=None,
        location="front",
        ticks="outside",
        all_edges=True,
        corner_factor=2,
        fmt=None,
        minor_ticks=True,
        padding=0.1,
        use_3d_text=False,
    )

    p.add_mesh(
        thresholded1,
        scalars="target",
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        scalar_bar_args=sargs1,
    )
    # This is a redundant mesh to create square mesh grid in all plots. Hence all values are NaNs and the mesh is hidden
    p.add_mesh(
        grid1,
        scalars=np.zeros([10000]) * np.nan,
        opacity=0.0,
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        show_scalar_bar=False,
    )

    light = pv.Light(position=camera_pos, focal_point=focal_point, color="white")
    light.intensity = 0.3
    p.add_light(light)  # Set light position and focal point
    p.camera_position = "xy"
    # p.camera.tight()
    p.camera.Zoom(1.2)

    p.subplot(0, 1)
    p.show_grid(
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=False,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=False,
        bold=True,
        font_size=15,
        font_family="courier",
        color=None,
        xtitle="X",
        ytitle="Y",
        n_xlabels=3,
        n_ylabels=5,
        use_2d=True,
        grid=None,
        location="front",
        ticks="outside",
        all_edges=True,
        corner_factor=2,
        fmt=None,
        minor_ticks=True,
        padding=0.1,
        use_3d_text=False,
    )
    p.add_mesh(
        thresholded2,
        scalars="pred",
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        scalar_bar_args=sargs2,
        clim=[np.min(target), np.max(target)]
    )  # , clim=[np.min(target), np.max(target)]
    p.add_mesh(
        grid2,
        scalars=np.zeros([10000]) * np.nan,
        opacity=0.0,
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        show_scalar_bar=False,
    )

    light = pv.Light(position=camera_pos, focal_point=focal_point, color="white")
    light.intensity = 0.3
    p.add_light(light)  # Set light position and focal point
    p.camera_position = "xy"
    # p.camera.tight()
    p.camera.Zoom(1.2)

    p.subplot(0, 2)
    p.show_grid(
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=False,
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=False,
        bold=True,
        font_size=15,
        font_family="courier",
        color=None,
        xtitle="X",
        ytitle="Y",
        n_xlabels=3,
        n_ylabels=5,
        use_2d=True,
        grid=None,
        location="front",
        ticks="outside",
        all_edges=True,
        corner_factor=2,
        fmt=None,
        minor_ticks=True,
        padding=0.1,
        use_3d_text=False,
    )
    p.add_mesh(
        thresholded3,
        scalars="error",
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        scalar_bar_args=sargs3,
    )
    p.add_mesh(
        grid3,
        scalars=np.zeros([10000]) * np.nan,
        opacity=0.0,
        cmap="inferno",
        render_points_as_spheres=False,
        point_size=9.0,
        show_scalar_bar=False,
    )

    # Adjust the light source
    light = pv.Light(position=camera_pos, focal_point=focal_point, color="white")
    light.intensity = 0.3
    p.add_light(light)  # Set light position and focal point
    p.camera_position = "xy"
    # p.camera.tight()
    p.camera.Zoom(1.2)
    # p.reset_camera_clipping_range()
    p.link_views()

    if save_plot:
        p.save_graphic(filename + ".svg")
        # p.show(interactive=True)
        p.close()
    else:
        p.show(interactive=True)


"""
Class containing individual sub-plot methods for loss
"""


class PlotLoss:
    def __init__(self, filename):
        self.filename = filename

    def plot_loss_target(self, batch_mse_loss, batch_err):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        # ax1 = fig.add_subplot(121)
        line1 = ax1.plot(batch_mse_loss, color="red", label="mse loss")
        line2 = ax1.plot(batch_err, color="blue", label="L2 rel error")
        ax1.set_xlabel("epochs")
        ax1.set_yscale("log")
        ax1.grid(True, which="major", linestyle="-.", linewidth=1.0, alpha=0.5)
        ax1.minorticks_on()
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.filename)
        # plt.show()

    def plot_loss_source(self, epoch_loss, epoch_err):
        fig, ax1 = plt.subplots(figsize=(12, 8))
        line1 = ax1.plot(epoch_loss, color="red", label="mse loss")
        line2 = ax1.plot(epoch_err, color="blue", label="L2 rel error")
        ax1.set_xlabel("epochs")
        ax1.set_yscale("log")
        ax1.grid(True, which="major", linestyle="-.", linewidth=1.0, alpha=0.5)
        ax1.minorticks_on()
        ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.filename)
