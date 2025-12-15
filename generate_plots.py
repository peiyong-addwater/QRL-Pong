# Python script which processes the csv data in runs_scalars_csv folder and generates plots
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science','nature'])
# set legnd font size globally
plt.rcParams['legend.fontsize'] = 10
# set axes label font size globally
plt.rcParams['axes.labelsize'] = 12
# set title font size globally
plt.rcParams['axes.titlesize'] = 12
# set the legend location globally: upper left
plt.rcParams['legend.loc'] = 'upper left'

# Plot settings
# Whether to smooth individual runs when plotting (for shading)
SMOOTH_INDIVIDUAL_RUNS = True
# Whether to plot min-max shading for runs
# set to true to disable individual run plotting
PLOT_MIN_MAX_SHADE = True
# Color palette
# Classical baseline: Black-dashed
CLASSICAL_COLOR = 'gray'
SIX_DIFFERNT_COLORS = sns.color_palette(["#E63946", "#F4A261", "#2A9D8F", "#1D3557", "#9D4EDD", "#06D6A0"])
SHADE_ALPHA = 0.1  # transparency for shading or individual runs
# line markers
# layer markers
LAYER_MARKERS = ['o', 's', 'D', '^', 'v', '+']
# model markers
MODEL_MARKERS = {
    'quantum_separable': 'o',
    'quantum_entangled': 's',
    'quantum_entangled_trainable_rzz': 'd',
    'classical_64': '*',
    'classical_128': '1',
    'classical_256': '2',
    'classical_336': '3',
    'classical_4096': '4'
}

# For data processing
UNIFORM_STEPS = np.linspace(0, 1e7, 2000)
EMA_SPAN = 100  # Span for exponential moving average smoothing

# Figure save dpi and size
FIG_DPI = 300
FIG_SIZE = (6, 4)

# file paths
STAT_FOLDER = Path('runs_scalars_csv')
PLOTS_FOLDER = Path('plots')
PLOTS_FOLDER.mkdir(exist_ok=True)
CSV_FILES = list(STAT_FOLDER.glob('*.csv'))
print(f"Found {len(CSV_FILES)} csv files to process.")

# csv filename formats:
# Classical baselines: Pong1PCB2L<64|128|256|336|4096>P__seed_<0|1|2|3|4|5|6|7|8|9>_<timestamp>.csv
# Quantum model stats: Pong1PQFM_XObs_<entangled|entangled_trainable_rzz|separable>_QLayers_<1|2|3|4|5|6>___seed_<0|1|2|3|4|5|6|7|8|9>_<timestamp>.csv
# Group the files based on model type, number of layers
res_groups = {}
res_groups['classical_64'] = []
res_groups['classical_128'] = []
res_groups['classical_256'] = []
res_groups['classical_336'] = []
res_groups['classical_4096'] = []
res_groups['quantum_entangled'] = {}
res_groups['quantum_entangled_trainable_rzz'] = {}
res_groups['quantum_separable'] = {}

classical_label_and_titles = {
    'classical_64': 'Classical Baseline 64 Params',
    'classical_128': 'Classical Baseline 128 Params',
    'classical_256': 'Classical Baseline 256 Params',
    'classical_336': 'Classical Baseline 336 Params',
    'classical_4096': 'Classical Baseline 4096 Params',
}
quantum_label_and_titles = {
    'quantum_separable': 'Quantum Separable',
    'quantum_entangled': 'Quantum CZ Entangled',
    'quantum_entangled_trainable_rzz': 'Quantum IsingZZ Entangled',
}

for csv_file in CSV_FILES:
    filename = csv_file.stem
    if 'Pong1PCB2L64P' in filename:
        res_groups['classical_64'].append(csv_file)
    elif 'Pong1PCB2L128P' in filename:
        res_groups['classical_128'].append(csv_file)
    elif 'Pong1PCB2L256P' in filename:
        res_groups['classical_256'].append(csv_file)
    elif 'Pong1PCB2L336P' in filename:
        res_groups['classical_336'].append(csv_file)
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

def load_csv_as_df(csv_file):
    """
    Load the given csv file into a pandas dataframe.
    """
    df = pd.read_csv(csv_file)
    return df

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
    interp_func = Akima1DInterpolator(df_smoothed['step'], df_smoothed['value'], extrapolate=None)
    resampled_values = interp_func(UNIFORM_STEPS)
    df_resampled = pd.DataFrame({'step': UNIFORM_STEPS, 'value': resampled_values})
    return df_resampled

def mean_across_runs(dfs, span=50):
    """
    Given a list of dataframes (each for a run), smooth and resample them,
    then compute the mean across runs.
    """
    resampled_dfs = [smooth_and_resample(df, span=span) for df in dfs]
    mean_values = np.mean([df['value'].values for df in resampled_dfs], axis=0)
    df_mean = pd.DataFrame({'step': UNIFORM_STEPS, 'value': mean_values})
    return df_mean

def minmax_across_runs(dfs, span=50):
    """
    Given a list of dataframes (each for a run), smooth and resample them,
    then compute the min and max across runs.
    """
    resampled_dfs = [smooth_and_resample(df, span=span) for df in dfs]
    min_values = np.min([df['value'].values for df in resampled_dfs], axis=0)
    max_values = np.max([df['value'].values for df in resampled_dfs], axis=0)
    df_minmax = pd.DataFrame({'step': UNIFORM_STEPS, 'min': min_values, 'max': max_values})
    return df_minmax

def plot_quantum_classical_comparison():
    """
    Generate plots comparing quantum models with all classical baselines.
    """
    quantum_groups = ['quantum_separable', 'quantum_entangled', 'quantum_entangled_trainable_rzz']
    classical_groups = list(classical_label_and_titles.keys())
    for classical_group in classical_groups:
        for quantum_group in quantum_groups:
            fig = plt.figure(figsize=FIG_SIZE)
            # Quantum models
            dfs = res_groups[quantum_group]
            # sort layers by integer value
            dfs = dict(sorted(dfs.items(), key=lambda item: int(item[0])))
            for idx, (n_layers, files) in enumerate(dfs.items()):
                run_dfs = [select_step_value(load_csv_as_df(f), '0-Episodic-Stats/episodic_return') for f in files]
                df_mean = mean_across_runs(run_dfs, span=EMA_SPAN)
                plt.plot(df_mean['step'], df_mean['value'], label=f'{quantum_label_and_titles[quantum_group]}; {n_layers} Q-Layers', color=SIX_DIFFERNT_COLORS[idx], marker = LAYER_MARKERS[idx], markevery=0.1)
                if PLOT_MIN_MAX_SHADE:
                    df_minmax = minmax_across_runs(run_dfs, span=EMA_SPAN)
                    plt.fill_between(df_minmax['step'], df_minmax['min'], df_minmax['max'], alpha=SHADE_ALPHA, color=SIX_DIFFERNT_COLORS[idx])
                else:
                    # instead of fill between min and max, plot all runs with low alpha and same color, for better visualization
                    for run_df in run_dfs:
                        if SMOOTH_INDIVIDUAL_RUNS:
                            df_smoothed = time_series_exponential_moving_average(run_df, span=EMA_SPAN)
                            plt.plot(df_smoothed['step'], df_smoothed['value'], color=SIX_DIFFERNT_COLORS[idx], alpha=SHADE_ALPHA)
                        else:
                            plt.plot(run_df['step'], run_df['value'], color=SIX_DIFFERNT_COLORS[idx], alpha=SHADE_ALPHA)
            # Classical baselines
            files = res_groups[classical_group]
            run_dfs = [select_step_value(load_csv_as_df(f), '0-Episodic-Stats/episodic_return') for f in files]
            df_mean = mean_across_runs(run_dfs, span=EMA_SPAN)
            if PLOT_MIN_MAX_SHADE:
                df_minmax = minmax_across_runs(run_dfs, span=EMA_SPAN)
                plt.fill_between(df_minmax['step'], df_minmax['min'], df_minmax['max'], alpha=SHADE_ALPHA, color=CLASSICAL_COLOR)
            plt.plot(df_mean['step'], df_mean['value'], label=classical_label_and_titles[classical_group], linestyle='--', color=CLASSICAL_COLOR, marker=MODEL_MARKERS[classical_group], markevery=0.1)
            if not PLOT_MIN_MAX_SHADE:
                # instead of fill between min and max, plot all runs with low alpha and same color, for better visualization
                for run_df in run_dfs:
                    if SMOOTH_INDIVIDUAL_RUNS:
                        df_smoothed = time_series_exponential_moving_average(run_df, span=EMA_SPAN)
                        plt.plot(df_smoothed['step'], df_smoothed['value'], color=CLASSICAL_COLOR, alpha=SHADE_ALPHA)
                    else:
                        plt.plot(run_df['step'], run_df['value'], color=CLASSICAL_COLOR, alpha=SHADE_ALPHA)
            plt.xlabel('Training Steps')
            plt.ylabel('Averaged Episodic Return')
            plt.title(f'Episodic Return: {quantum_label_and_titles[quantum_group]} vs {classical_label_and_titles[classical_group]}')
            plt.legend()
            plt.grid(True)
            plt.savefig(PLOTS_FOLDER / f'episodic_return_{quantum_group}_vs_{classical_group}.pdf', dpi=FIG_DPI)
            plt.close()
    return None

if __name__ == "__main__":

    # 1~4. Compare quantum models w.r.t. classical baselines
    plot_quantum_classical_comparison()
    # 5. Compare entangled quantum models (two types) w.r.t. separable models for each number of layers
    dfs = res_groups['quantum_separable']
    for n_layers in dfs.keys():
        idx = int(n_layers) - 1  # zero-based index
        fig = plt.figure(figsize=FIG_SIZE)
        # Separable model
        files = res_groups['quantum_separable'].get(n_layers, [])
        if files:
            run_dfs = [select_step_value(load_csv_as_df(f), '0-Episodic-Stats/episodic_return') for f in files]
            df_mean = mean_across_runs(run_dfs, span=EMA_SPAN)
            plt.plot(df_mean['step'], df_mean['value'], label=f'Quantum Separable; {n_layers} Q-Layers', color=SIX_DIFFERNT_COLORS[0], marker=MODEL_MARKERS['quantum_separable'], markevery=0.1)
            if PLOT_MIN_MAX_SHADE:
                df_minmax = minmax_across_runs(run_dfs, span=EMA_SPAN)
                plt.fill_between(df_minmax['step'], df_minmax['min'], df_minmax['max'], alpha=SHADE_ALPHA, color=SIX_DIFFERNT_COLORS[0])
            else:
                # instead of fill between min and max, plot all runs with low alpha and same color, for better visualization
                for run_df in run_dfs:
                    if SMOOTH_INDIVIDUAL_RUNS:
                        df_smoothed = time_series_exponential_moving_average(run_df, span=EMA_SPAN)
                        plt.plot(df_smoothed['step'], df_smoothed['value'], color=SIX_DIFFERNT_COLORS[0], alpha=SHADE_ALPHA)
                    else:
                        plt.plot(run_df['step'], run_df['value'], color=SIX_DIFFERNT_COLORS[0], alpha=0.05)
        # Entangled model
        files = res_groups['quantum_entangled'].get(n_layers, [])
        if files:
            run_dfs = [select_step_value(load_csv_as_df(f), '0-Episodic-Stats/episodic_return') for f in files]
            df_mean = mean_across_runs(run_dfs, span=EMA_SPAN)
            plt.plot(df_mean['step'], df_mean['value'], label=f'Quantum CZ Entangled; {n_layers} Q-Layers', color=SIX_DIFFERNT_COLORS[2], marker=MODEL_MARKERS['quantum_entangled'], markevery=0.1)
            if PLOT_MIN_MAX_SHADE:
                df_minmax = minmax_across_runs(run_dfs, span=EMA_SPAN)
                plt.fill_between(df_minmax['step'], df_minmax['min'], df_minmax['max'], alpha=SHADE_ALPHA, color=SIX_DIFFERNT_COLORS[2])
            else:
                # instead of fill between min and max, plot all runs with low alpha and same color, for better visualization
                for run_df in run_dfs:
                    if SMOOTH_INDIVIDUAL_RUNS:
                        df_smoothed = time_series_exponential_moving_average(run_df, span=EMA_SPAN)
                        plt.plot(df_smoothed['step'], df_smoothed['value'], color=SIX_DIFFERNT_COLORS[2], alpha=SHADE_ALPHA)
                    else:
                        plt.plot(run_df['step'], run_df['value'], color=SIX_DIFFERNT_COLORS[2], alpha=0.05)
        # Entangled trainable rzz model
        files = res_groups['quantum_entangled_trainable_rzz'].get(n_layers, [])
        if files:
            run_dfs = [select_step_value(load_csv_as_df(f), '0-Episodic-Stats/episodic_return') for f in files]
            df_mean = mean_across_runs(run_dfs, span=EMA_SPAN)
            plt.plot(df_mean['step'], df_mean['value'], label=f'Quantum IsingZZ Entangled; {n_layers} Q-Layers', color=SIX_DIFFERNT_COLORS[4], marker=MODEL_MARKERS['quantum_entangled_trainable_rzz'], markevery=0.1)
            if PLOT_MIN_MAX_SHADE:
                df_minmax = minmax_across_runs(run_dfs, span=EMA_SPAN)
                plt.fill_between(df_minmax['step'], df_minmax['min'], df_minmax['max'], alpha=SHADE_ALPHA, color=SIX_DIFFERNT_COLORS[4])
            else:
                # instead of fill between min and max, plot all runs with low alpha and same color, for better visualization
                for run_df in run_dfs:
                    if SMOOTH_INDIVIDUAL_RUNS:
                        df_smoothed = time_series_exponential_moving_average(run_df, span=EMA_SPAN)
                        plt.plot(df_smoothed['step'], df_smoothed['value'], color=SIX_DIFFERNT_COLORS[4], alpha=SHADE_ALPHA)
                    else:
                        plt.plot(run_df['step'], run_df['value'], color=SIX_DIFFERNT_COLORS[4], alpha=0.05)
        plt.xlabel('Training Steps')
        plt.ylabel('Averaged Episodic Return')
        plt.legend()
        plt.title(f'Episodic Return Comparison for {n_layers} Q-Layers')
        plt.grid(True)
        plt.savefig(PLOTS_FOLDER / f'episodic_return_quantum_only_comparison_{n_layers}_layers.pdf', dpi=FIG_DPI)
        plt.close()