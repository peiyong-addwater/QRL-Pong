# Python script which processes the csv data in runs_scalars_csv folder and generates plots
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science','nature'])

# For data processing
UNIFORM_STEPS = np.linspace(0, 1e7, 2000)

# file paths
STAT_FOLDER = Path('runs_scalars_csv')
PLOTS_FOLDER = Path('plots')
PLOTS_FOLDER.mkdir(exist_ok=True)
CSV_FILES = list(STAT_FOLDER.glob('*.csv'))
print(f"Found {len(CSV_FILES)} csv files to process.")

# csv filename formats:
# Classical baselines: Pong1PCB2L<64|4096>P__seed_<0|1|2|3|4|5|6|7|8|9>_<timestamp>.csv
# Quantum model stats: Pong1PQFM_XObs_<entangled|entangled_trainable_rzz|separable>_QLayers_<1|2|3|4|5|6>___seed_<0|1|2|3|4|5|6|7|8|9>_<timestamp>.csv
# Group the files based on model type, number of layers
res_groups = {}
res_groups['classical_64'] = []
res_groups['classical_4096'] = []
res_groups['quantum_entangled'] = {}
res_groups['quantum_entangled_trainable_rzz'] = {}
res_groups['quantum_separable'] = {}

for csv_file in CSV_FILES:
    filename = csv_file.stem
    if 'Pong1PCB2L64P' in filename:
        res_groups['classical_64'].append(csv_file)
    elif 'Pong1PCB2L4096P' in filename:
        res_groups['classical_4096'].append(csv_file)
    elif 'Pong1PQFM_XObs_entangled_trainable_rzz' in filename:
        n_layers = filename.split('QLayers_')[1].split('___seed_')[0]
        if n_layers not in res_groups['quantum_entangled_trainable_rzz']:
            res_groups['quantum_entangled_trainable_rzz'][n_layers] = []
        res_groups['quantum_entangled_trainable_rzz'][n_layers].append(csv_file)
    elif 'Pong1PQFM_XObs_entangled' in filename:
        n_layers = filename.split('QLayers_')[1].split('___seed_')[0]
        if n_layers not in res_groups['quantum_entangled']:
            res_groups['quantum_entangled'][n_layers] = []
        res_groups['quantum_entangled'][n_layers].append(csv_file)
    elif 'Pong1PQFM_XObs_separable' in filename:
        n_layers = filename.split('QLayers_')[1].split('___seed_')[0]
        if n_layers not in res_groups['quantum_separable']:
            res_groups['quantum_separable'][n_layers] = []
        res_groups['quantum_separable'][n_layers].append(csv_file)

# For each csv, there are four columns: step, tag, value, run
# For `tag`, there are the following unique values:
# '0-Episodic-Stats/episodic_length' '0-Episodic-Stats/episodic_return'
# '1-Training-Losses/approx_kl' '1-Training-Losses/clipfrac'
# '1-Training-Losses/entropy' '1-Training-Losses/explained_variance'
# '1-Training-Losses/policy_loss' '1-Training-Losses/value_loss'
# '2-Training-Stats/SPS' '2-Training-Stats/learning_rate'
# We only need the follwing tags for plotting:
PLOT_TAGS = [
    '0-Episodic-Stats/episodic_return',
    '0-Episodic-Stats/episodic_length',
    '1-Training-Losses/policy_loss',
    '1-Training-Losses/value_loss',
    '1-Training-Losses/explained_variance',
    '1-Training-Losses/entropy',
    '1-Training-Losses/clipfrac',
    '1-Training-Losses/approx_kl'
]

TAGS_AND_LABELS_TITLES = {
    '0-Episodic-Stats/episodic_return': 'Episodic Return',
    '0-Episodic-Stats/episodic_length': 'Episodic Length',
    '1-Training-Losses/policy_loss': 'Policy Loss',
    '1-Training-Losses/value_loss': 'Value Loss',
    '1-Training-Losses/explained_variance': 'Explained Variance',
    '1-Training-Losses/entropy': 'Entropy',
    '1-Training-Losses/clipfrac': 'Clip Fraction',
    '1-Training-Losses/approx_kl': 'Approx. KL Divergence'
}

def select_step_value(df, tag):
    """
    Return the paris of (step, value) for the given tag from the dataframe.
    """
    df_tag = df[df['tag'] == tag]
    return df_tag[['step', 'value']]

def time_series_exponential_moving_average(df, span):
    """
    Apply exponential moving average smoothing to the 'value' column of the dataframe.
    """
    df_smoothed = df.copy()
    df_smoothed['value'] = df_smoothed['value'].ewm(span=span, adjust=False).mean()
    return df_smoothed

def smooth_and_resample(df, span=50):
    """
    Smooth the time series data using EMA and resample to uniform steps.
    """
    df_smoothed = time_series_exponential_moving_average(df, span=span)
    interp_func = interp1d(df_smoothed['step'], df_smoothed['value'], kind='linear', fill_value="extrapolate")
    resampled_values = interp_func(UNIFORM_STEPS)
    df_resampled = pd.DataFrame({'step': UNIFORM_STEPS, 'value': resampled_values})
    return df_resampled

# test with one csv file
sample_df = pd.read_csv(CSV_FILES[0])
print("Sample CSV data:")
print(sample_df.head())
# print unique tags
print("Unique tags in sample CSV:")
print(sample_df['tag'].unique())
# select sample for episodic_return
sample_data = select_step_value(sample_df, '0-Episodic-Stats/episodic_return')
print("Sample episodic_return data:")
print(sample_data.head())
# plot sample data
plt.figure(figsize=(5, 3))
# original data as a very light line
sns.lineplot(data=sample_data, x='step', y='value', alpha=0.3)
# plot the EMA
smoothed_data = smooth_and_resample(sample_data, span=50)
sns.lineplot(data=smoothed_data, x='step', y='value', label='EMA (span=50, Smoothed and Resampled)', color='red')
plt.title('Episodic Return Over Training Steps')
plt.xlabel('Step')
plt.ylabel('Episodic Return')
plt.grid()
# save as test plot
plt.savefig(PLOTS_FOLDER / 'sample_plot.png', dpi=300)
plt.close()