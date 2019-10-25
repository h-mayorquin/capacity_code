import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis_functions import calculate_angle_from_history, calculate_winning_pattern_from_distances
from analysis_functions import calculate_patterns_timings


class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



def plot_weight_matrix(manager, one_hypercolum=True, ax=None, vmin=None, title=True, transpose=False):
    with sns.axes_style("whitegrid", {'axes.grid': False}):

        w = manager.nn.w

        if one_hypercolum:
            w = w[:manager.nn.minicolumns, :manager.nn.minicolumns]

        # aux_max = np.max(np.abs(w))
        norm = MidpointNormalize(midpoint=0)
        cmap = matplotlib.cm.RdBu_r

        if ax is None:
            # sns.set_style("whitegrid", {'axes.grid': False})
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if transpose:
            matrix_to_plot = w.T
        else:
            matrix_to_plot = w
        im = ax.imshow(matrix_to_plot, cmap=cmap, interpolation='None', norm=norm, vmin=vmin)

        if title:
            ax.set_title('w connectivity')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = ax.get_figure().colorbar(im, ax=ax, cax=cax)

    return ax


def plot_persistent_matrix(manager, ax=None):
    with sns.axes_style("whitegrid", {'axes.grid': False}):
        title = r'$T_{persistence}$'

        cmap = matplotlib.cm.Reds_r
        cmap.set_bad(color='white')
        cmap.set_under('black')

        if ax is None:
            # sns.set_style("whitegrid", {'axes.grid': False})
            fig = plt.figure()
            ax = fig.add_subplot(111)

        im = ax.imshow(manager.T, cmap=cmap, interpolation='None', vmin=0.0)
        ax.set_title(title)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.get_figure().colorbar(im, ax=ax, cax=cax)


def plot_network_activity(manager, recall=True, cmap=None, ax=None, title=True, time_y=True):
    if recall:
        T_total = manager.T_recall_total
    else:
        T_total = manager.T_training_total

    history = manager.history
    # Get the angles
    patterns_dic = manager.patterns_dic

    # Plot
    sns.set_style("whitegrid", {'axes.grid': False})
    if cmap is None:
        cmap = matplotlib.cm.binary

    if title:
        title = 'Unit activation'
    else:
        title = ''

    if ax is None:
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)

    else:
        fig = ax.figure

    if time_y:
        to_plot = history['o']
        origin = 'upper'
        x_label = 'Units'
        y_label = 'Time (s)'
        extent = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]

    else:
        to_plot = history['o'].T
        origin = 'lower'
        y_label = 'Units'
        x_label = 'Time (s)'
        extent = [0, T_total, 0, manager.nn.minicolumns * manager.nn.hypercolumns]

    im = ax.imshow(to_plot, aspect='auto', origin=origin, interpolation='None', cmap=cmap,
                   vmax=1, vmin=0, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if ax is None:
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
        fig.colorbar(im, cax=cbar_ax)

    return ax

def plot_network_activity_angle(manager, recall=True, cmap=None, ax=None, title=True, time_y=True):
    if recall:
        T_total = manager.T_recall_total
    else:
        T_total = manager.T_training_total

    history = manager.history
    # Get the angles
    angles = calculate_angle_from_history(manager)
    patterns_dic = manager.patterns_dic
    n_patters = len(patterns_dic)
    # Plot
    sns.set_style("whitegrid", {'axes.grid': False})

    if cmap is None:
        cmap = 'plasma'

    if title:
        title1 = 'Unit activation'
        title2 = 'Angles with stored'
    else:
        title1 = ''
        title2 = ''

    if ax is None:
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        ax1, ax2 = ax
        fig = ax1.figure

    if time_y:
        extent1 = [0, manager.nn.minicolumns * manager.nn.hypercolumns, T_total, 0]
        extent2 = [0, n_patters, T_total, 0]

        im1 = ax1.imshow(history['o'], aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent1)
        ax1.set_title(title1)

        ax1.set_xlabel('Units')
        ax1.set_ylabel('Time (s)')

        im2 = ax2.imshow(angles, aspect='auto', interpolation='None', cmap=cmap, vmax=1, vmin=0, extent=extent2)
        ax2.set_title(title2)
        ax2.set_xlabel('Patterns')

    else:
        extent1 = [0, T_total, 0, manager.nn.minicolumns * manager.nn.hypercolumns]
        extent2 = [0, T_total, 0, n_patters]

        im1 = ax1.imshow(history['o'].T, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0, extent=extent1)
        ax1.set_title(title1)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Units')

        im2 = ax2.imshow(angles.T, aspect='auto', origin='lower', cmap=cmap, vmax=1, vmin=0, extent=extent2)
        ax2.set_title(title2)
        ax2.set_ylabel('Patterns')
        ax2.set_xlabel('Time (s)')

    if ax is None:
        fig.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.12, 0.05, 0.79])
        fig.colorbar(im1, cax=cbar_ax)



    return [ax1, ax2]