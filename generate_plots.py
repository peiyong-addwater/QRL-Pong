# Python script which processes the csv data in runs_scalars_csv folder and generates plots
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def select_step_value(df, tag):
    """
    Return the paris of (step, value) for the given tag from the dataframe.
    """
    df_tag = df[df['tag'] == tag]
    return df_tag[['step', 'value']]

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
plt.figure(figsize=(10, 6))
sns.lineplot(data=sample_data, x='step', y='value')
plt.title('Sample Episodic Return over Steps')
plt.xlabel('Step')
plt.ylabel('Episodic Return')
plt.grid()
# save as test plot
plt.savefig(PLOTS_FOLDER / 'sample_episodic_return.png')
plt.close()