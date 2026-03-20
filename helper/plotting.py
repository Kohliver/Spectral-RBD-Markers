#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:07:12 2023

@author: okohl

Functions aiding plotting of the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
import seaborn as sns


def crop_psd(psd,f,f_range=[2,45]):

    f_mask = np.logical_and(f >= f_range[0],f <= f_range[1])
    cropped_psd = psd[:,f_mask]
    cropped_f = f[f_mask]
    
    return cropped_f, cropped_psd


def data_to_longformat(psd,f,group_list):

    nSub = psd.shape[0]
    nFreq = psd.shape[1]
    
    long_psd = psd.flatten()
    long_f = np.tile(f,nSub)
    long_group = np.repeat(group_list,nFreq)
    
    plot_in = np.vstack([long_psd,long_f,long_group]).T
    plot_in = pd.DataFrame(plot_in, columns=['psd', 'f', 'group'])
    
    return plot_in


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def prepare_plotting(inverse_cmp=False):
    
    group_labels = [r'HC$_{PD}$','PD',r'HC$_{RBD}$','RBD']
    
    if inverse_cmp == True:
        group_cmaps = ['Greys_r', 'Blues_r', 'Greys_r', 'Reds_r', 'Purples_r', "Oranges_r"]
        
        cropped_cmaps = []
        for color in group_cmaps:
            cmap = plt.get_cmap(color)
            cropped_cmaps.append((truncate_colormap(cmap, 0.05, 0.9)))
    else:
        group_cmaps = ['Greys', 'Blues', 'Greys', 'Reds', 'Purples', "Oranges"]
        
        cropped_cmaps = []
        for color in group_cmaps:
            cmap = plt.get_cmap(color)
            cropped_cmaps.append((truncate_colormap(cmap, 0.1, 0.95)))

    group_colors = []
    for color in group_cmaps:
        group_colors.append(get_cmap(color)(.65))
                
    return group_labels, group_colors, group_cmaps, cropped_cmaps


def psd_plot(plot_in,group_colors,group_labels,ax):
    
    # Line Plot
    sns.lineplot(data=plot_in, x="f", y="psd", hue="group",
                 linewidth=3, palette=group_colors,
                 ax=ax)
    
    # Make Axes pretty
    ax.set_xlim([1, 46])
    ax.set_xlabel('Frequency (Hz)',fontsize=16)
    ax.set_ylabel('Power (a.u.)', fontsize=16,labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.ticklabel_format(scilimits=(-1,10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    # Legend
    ax.legend(frameon=False, fontsize=14,
              labelspacing=0.2,
              handletextpad=.5,
              handlelength=1)
    leg = ax.axes.get_legend()
    
    new_labels = group_labels
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    
    # Remove Box around plot
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
