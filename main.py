"""
Compute power spectral density (PSD) of raw data.

This app loads raw MEG/EEG data and computes PSD per channel type
(EEG/gradiometers/magnetometers) via Welch's method, saving TSV tables
and PSD plots.

Inputs:
    - mne: Path to raw MEG/EEG data file
    - fmin, fmax: Frequency range
    - average: Whether to average PSD estimates
    - tmin, tmax: Optional time range
    - n_fft, n_overlap, n_per_seg, window: Welch PSD parameters
    - reject_by_annotation: Whether to reject annotated segments
    - proj: Whether to apply SSP projectors

Outputs:
    - out_psd_eeg/psd.tsv, out_psd_grad/psd.tsv, out_psd_mag/psd.tsv: Per-channel-type PSD tables
    - out_figs/psd_computed.png: Computed PSD plot
    - out_figs/psd_mne.png: MNE PSD plot
    - product.json: Metadata about the computed PSD
"""

# Copyright (c) 2026 brainlife.io
#
# Author: Guiomar Niso

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

# Standard imports
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import shared utilities
from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
    require_config_keys
)

# Set up matplotlib for headless execution
setup_matplotlib_backend()

# Ensure output directories exist
ensure_output_dirs('out_psd_eeg', 'out_psd_grad', 'out_psd_mag', 'out_figs')

# Load configuration
config = load_config()
require_config_keys(config, ['mne', 'fmin', 'fmax'])

# == LOAD DATA ==
fname = config['mne']
raw = mne.io.read_raw(fname)

# == GET CONFIG VALUES ==
fmin = config['fmin']
fmax = config['fmax']
# compute_psd's average kwarg expects 'mean' | 'median' | None (not a bool)
average = 'mean' if config['average'] else None

# Advanced parameters
tmin = config['tmin'] if config['tmin'] else None
tmax = config['tmax'] if config['tmax'] else None
n_fft = config['n_fft']
n_overlap = config['n_overlap']
n_per_seg = config['n_per_seg'] if config['n_per_seg'] else None
window = config['window']
reject_by_annotation = config['reject_by_annotation']
proj = config['proj']
n_jobs = 1
picks = None

# Dimensions: psd.shape (from Spectrum.get_data()): Nchannels x Nfreqs

# Types of channels in the data
# e.g. ['ecg', 'eog', 'grad', 'mag', 'eeg','misc', 'stim']
ch_types = np.unique(raw.get_channel_types())

# == COMPUTE PSD ==
if picks == None:

    # FIGURE 1: PSD manually computed
    # Number of subplots
    num_subplots = 0
    for i in ['grad', 'mag', 'eeg']:
        if i in ch_types: num_subplots = num_subplots + 1
    plt.figure(1)
    fig, axs = plt.subplots(num_subplots)
    fig.subplots_adjust(hspace=.5, wspace=.2)

    aa = 0

    if 'eeg' in ch_types:
        raw_eeg = raw.copy().pick('eeg')
        ch_eeg = raw_eeg.ch_names
        spectrum_eeg = raw_eeg.compute_psd(method='welch',
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window,
                            reject_by_annotation=reject_by_annotation, average=average,
                            picks='eeg', proj=proj, n_jobs=1, verbose=None)
        psd_welch_eeg, freqs_eeg = spectrum_eeg.get_data(return_freqs=True)
        # Convert power to dB scale: V^2/hz -> uV^2/Hz
        psd_welch_eeg = 10*(np.log10(psd_welch_eeg*1e6**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_eeg, index=ch_eeg, columns=freqs_eeg)
        df_psd.index.name = 'channels'
        df_psd.columns.name = 'freqs'
        eeg_tsv_path = os.path.join('out_psd_eeg', 'psd.tsv')
        df_psd.to_csv(eeg_tsv_path, sep='\t')

        if num_subplots == 1:
            # Figure
            axs.plot(freqs_eeg, psd_welch_eeg.transpose(), zorder=1)
            axs.set_xlim(xmin=0, xmax=max(freqs_eeg))
            axs.set_xlabel('Frequency (Hz)')
            axs.set_ylabel('uV^2/Hz [dB]')
            axs.set_title('PSD - EEG')
            axs.grid(linestyle=':')

        elif num_subplots > 1:
            # Figure
            axs[aa].plot(freqs_eeg, psd_welch_eeg.transpose(), zorder=1)
            axs[aa].set_xlim(xmin=0, xmax=max(freqs_eeg))
            axs[aa].set_xlabel('Frequency (Hz)')
            axs[aa].set_ylabel('uV^2/Hz [dB]')
            axs[aa].set_title('PSD - EEG')
            axs[aa].grid(linestyle=':')
            aa = aa + 1

    if 'grad' in ch_types:
        raw_grad = raw.copy().pick('grad')
        ch_grad = raw_grad.ch_names
        spectrum_grad = raw_grad.compute_psd(method='welch',
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window,
                            reject_by_annotation=reject_by_annotation, average=average,
                            picks='grad', proj=proj, n_jobs=n_jobs, verbose=None)
        psd_welch_grad, freqs_grad = spectrum_grad.get_data(return_freqs=True)
        # Convert power to dB scale: (T/m)^2/hz -> (fT/cm)^2/Hz
        psd_welch_grad = 10*(np.log10(psd_welch_grad*1e13**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_grad, index=ch_grad, columns=freqs_grad)
        df_psd.index.name = 'channels'
        df_psd.columns.name = 'freqs'
        grad_tsv_path = os.path.join('out_psd_grad', 'psd.tsv')
        df_psd.to_csv(grad_tsv_path, sep='\t')

        if num_subplots == 1:
            # Figure
            axs.plot(freqs_grad, psd_welch_grad.transpose(), zorder=1)
            axs.set_xlim(xmin=0, xmax=max(freqs_grad))
            axs.set_xlabel('Frequency (Hz)')
            axs.set_ylabel('(fT/cm)^2/Hz [dB]')
            axs.set_title('PSD - Gradieometers')
            axs.grid(linestyle=':')

        elif num_subplots > 1:
            # Figure
            axs[aa].plot(freqs_grad, psd_welch_grad.transpose(), zorder=1)
            axs[aa].set_xlim(xmin=0, xmax=max(freqs_grad))
            axs[aa].set_xlabel('Frequency (Hz)')
            axs[aa].set_ylabel('(fT/cm)^2/Hz [dB]')
            axs[aa].set_title('PSD - Gradieometers')
            axs[aa].grid(linestyle=':')
            aa = aa + 1

    if 'mag' in ch_types:
        raw_mag = raw.copy().pick('mag')
        ch_mag = raw_mag.ch_names
        spectrum_mag = raw_mag.compute_psd(method='welch',
                            fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
                            n_fft=n_fft, n_overlap=n_overlap, n_per_seg=n_per_seg, window=window,
                            reject_by_annotation=reject_by_annotation, average=average,
                            picks='mag', proj=proj, n_jobs=n_jobs, verbose=None)
        psd_welch_mag, freqs_mag = spectrum_mag.get_data(return_freqs=True)
        # Convert power to dB scale: T^2/hz -> fT^2/Hz
        psd_welch_mag = 10*(np.log10(psd_welch_mag*1e15**2))

        # Save to TSV file
        df_psd = pd.DataFrame(psd_welch_mag, index=ch_mag, columns=freqs_mag)
        df_psd.index.name = 'channels'
        df_psd.columns.name = 'freqs'
        mag_tsv_path = os.path.join('out_psd_mag', 'psd.tsv')
        df_psd.to_csv(mag_tsv_path, sep='\t')

        if num_subplots == 1:
            # Figure
            axs.plot(freqs_mag, psd_welch_mag.transpose(), zorder=1)
            axs.set_xlim(xmin=0, xmax=max(freqs_mag))
            axs.set_xlabel('Frequency (Hz)')
            axs.set_ylabel('fT^2/Hz [dB]')
            axs.set_title('PSD - Magnetometers')
            axs.grid(linestyle=':')

        elif num_subplots > 1:
            # Figure
            axs[aa].plot(freqs_mag, psd_welch_mag.transpose(), zorder=1)
            axs[aa].set_xlim(xmin=0, xmax=max(freqs_mag))
            axs[aa].set_xlabel('Frequency (Hz)')
            axs[aa].set_ylabel('fT^2/Hz [dB]')
            axs[aa].set_title('PSD - Magnetometers')
            axs[aa].grid(linestyle=':')
            aa = aa + 1

    # Save fig
    computed_path = os.path.join('out_figs', 'psd_computed.png')
    plt.savefig(computed_path)
    plt.close(fig)

# FIGURE 2: PSD computed with MNE function
spectrum = raw.compute_psd(method='welch', fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            proj=proj, n_fft=n_fft, n_overlap=n_overlap, window=window,
            n_jobs=n_jobs, verbose=None)
fig2 = spectrum.plot(dB=True, xscale='linear', ci='sd', ci_alpha=0.33,
            color='black', alpha=None, spatial_colors=True, sphere=None,
            average=False, show=False)
# Save fig
mne_path = os.path.join('out_figs', 'psd_mne.png')
fig2.savefig(mne_path)
plt.close(fig2)

# == CREATE PRODUCT.JSON ==
product_items = []
add_info_to_product(product_items, f'Computed PSD for channel types: {list(ch_types)}', 'success')
add_image_to_product(product_items, 'Computed PSD', filepath=computed_path)
add_image_to_product(product_items, 'MNE PSD', filepath=mne_path)
create_product_json(product_items)
