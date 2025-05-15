import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

def load_all_metrics(csv_folder: Path, prefix: str = "metrics_seed_", suffix: str = ".csv", seeds=range(10)) -> pd.DataFrame:
    """load all CSV-files and combine them to a dataframe"""
    all_dfs = []
    for seed in seeds:
        path = csv_folder / f"{prefix}{seed}{suffix}"
        df = pd.read_csv(path)
        df["random_seed"] = seed
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def plot_metric_grid(df: pd.DataFrame, metric: str, output_file: Path):
    """make subplots: black lines for each seed, red for mean"""
    photo_counts = sorted(df["photo_count"].unique())
    n = len(photo_counts)
    cols = 3
    rows = math.ceil(n / cols)

    all_plots_figure, subplots = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharex=True, sharey=True)
    subplots = subplots.flatten()

    for idx, photo_count in enumerate(photo_counts):
        subplot = subplots[idx]
        df_photo_count_subset = df[df["photo_count"] == photo_count]

        # single seed curves (black)
        for seed in sorted(df_photo_count_subset["random_seed"].unique()):
            seed_data_subset = df_photo_count_subset[df_photo_count_subset["random_seed"] == seed]
            subplot.plot(seed_data_subset["epoch"], seed_data_subset[metric], color="black", alpha=0.9)

        # mean (per epoch)
        mean_data = df_photo_count_subset.groupby("epoch")[metric].mean().reset_index()
        subplot.plot(mean_data["epoch"], mean_data[metric], color="red", linewidth=3)

        subplot.plot([], [], color="black", alpha=0.5, label="single seeds")
        subplot.plot([], [], color="red", linewidth=2, label="mean")

        subplot.legend(loc="best", fontsize=8)

        subplot.set_title(f"Photo count: {photo_count}")
        subplot.set_xlabel("Epoch")
        subplot.set_ylabel(metric.replace("_", " ").capitalize())
        subplot.grid(True)

    for j in range(len(photo_counts), len(subplots)):
        all_plots_figure.delaxes(subplots[j])

    all_plots_figure.suptitle(f"{metric.replace('_', ' ').capitalize()} vs Epochs\n(Black = Seeds, Red = Mean)", fontsize=16)
    all_plots_figure.tight_layout(rect=[0, 0.05, 1, 0.95])
    all_plots_figure.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    input_folder = Path("./")  # folder with metrics_seed_*.csv
    output_folder = Path("./plots")
    output_folder.mkdir(exist_ok=True)

    df_all = load_all_metrics(input_folder)

    plot_metric_grid(df_all, "error_rate", output_folder / "error_rate.png")
    plot_metric_grid(df_all, "valid_loss", output_folder / "valid_loss.png")
    plot_metric_grid(df_all, "train_loss", output_folder / "train_loss.png")