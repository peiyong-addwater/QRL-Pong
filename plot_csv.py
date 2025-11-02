# Plot CSV data as a heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


CSV_PATH = "cka_results.csv" # path to the csv file containing CKA results
OUTPUT_PATH = "cka_heatmap.png" # path to save the heatmap image

def plot_heatmap(csv_path, output_path):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_path, index_col=0)

    # Create a heatmap using seaborn
    # note: the csv has a huge number of rows/columns, so we set a large figure size
    plt.figure(figsize=(60, 48), dpi=600)
    # no annotation to avoid clutter
    sns.heatmap(data, annot=False, cmap="viridis")

    # Save the heatmap to a file
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    plot_heatmap(CSV_PATH, OUTPUT_PATH)