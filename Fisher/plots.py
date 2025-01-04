"""
Author: Varun Kumar
Date: Jan 3 2025
Purpose: model script for Fisher problem
Contact: varun_kumar2@brown.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PlotLoss:

    def __init__(self, filename):
        self.filename = filename

    def plot_loss(self, train_loss, train_error):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        line1 = ax1.plot(train_loss, color="red", label='train_loss')
        line2 = ax1.plot(train_error, color="blue", label='train_err')
        ax1.set_xlabel("epochs")
        ax1.set_yscale("log")
        ax1.grid(True, which='major', linestyle='-.', linewidth=1.0, alpha=0.5)
        ax1.minorticks_on()
        ax1.grid(True, which='minor', linestyle=':', linewidth=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.filename)
        # plt.show()
        plt.close()


def plot_Fisher_separate(tt, xx, outputs, filename):
    """
    Function to create individual results for Fisher test cases
    :param tt: meshgrid data 20x64
    :param xx: meshgrid data 20x64
    :param outputs: Output field data with all 3 components (target, pred, error) arranged as bs x 3
    :param filename: File name to save the plot
    :return: None
    """

    def format_func_y1(value, tick_number):
        return f'{value:.2f}'

    common_title = ['reference', 'prediction', 'absolute error']
    x_ticks = np.linspace(0, 1.0, num=5)
    x_ticks = np.round(x_ticks, 2)

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharex=True, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1]})

    # Plot the data in each subplot
    for j in range(3):
        ax = axs[j]
        color_data = outputs[:, :, j]
        if j == 0:
            cf = ax.contourf(xx, tt, color_data, cmap='inferno',
                             levels=100, vmin=np.min(outputs[:, :, 1]), vmax=np.max(outputs[:, :, 1]))  # Use your desired colormap, , vmin=np.min(outputs[0, :, :, 0]), vmax=np.max(outputs[0, :, :, 0])
        else:
            cf = ax.contourf(xx, tt, color_data, cmap='inferno', levels=100)
        ax.set_aspect('equal')
        ax.tick_params(axis='both', labelsize=14)

        ax.set_xlabel('x', fontsize=14, fontweight='normal')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xticks(x_ticks)

        if j == 1:
            ax.set_ylabel('time', fontsize=14, fontweight='normal')
            ax.tick_params(axis='both', labelsize=14)

        if j >= 1:  # Adding colorbars only to columns 2, 3 and 4
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.2)
            cbar = plt.colorbar(cf, cax=cax, orientation='vertical', format='%.3f',
                                ticks=np.linspace(np.round(outputs[:, :, j].min(), 3), np.round(outputs[:, :, j].max(), 3), num=6))

        plt.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelsize=14)
        # fig.text(0.0, 0.1*i, f'{common_ylabel[i]}', ha='center', va='center', rotation='vertical')
    # Adjust the spacing between plots
    plt.subplots_adjust(wspace=0.2, hspace=0.25)

    # Set common xlabel and ylabel (if needed)
    # fig.text(0.5, 0.04, 'X-axis Label', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    # plt.show()
    plt.close()
