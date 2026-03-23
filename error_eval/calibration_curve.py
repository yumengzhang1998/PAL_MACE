import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def calibration_curve(uncertainty, error, n_bins=20):

    # sort by uncertainty
    order = np.argsort(uncertainty)

    uncertainty = uncertainty[order]
    error = error[order]

    bins = np.array_split(np.arange(len(uncertainty)), n_bins)

    mean_unc = []
    mean_err = []

    for b in bins:
        mean_unc.append(np.mean(uncertainty[b]))
        mean_err.append(np.mean(error[b]))

    return np.array(mean_unc), np.array(mean_err)


def plot_calibration(df, unc_col, err_col, ax):

    unc = df[unc_col].values
    err = df[err_col].values

    # Spearman correlation
    rho, p = spearmanr(unc, err)

    # calibration bins
    mean_unc, mean_err = calibration_curve(unc, err)

    ax.scatter(mean_unc, mean_err, s=50)

    # ideal calibration line
    lim = max(mean_unc.max(), mean_err.max())
    ax.plot([0, lim], [0, lim], '--', color='black', label="Ideal")

    ax.set_xlabel("Predicted uncertainty")
    ax.set_ylabel("Observed error")

    ax.set_title(
        f"{unc_col} vs {err_col}\nSpearman = {rho:.3f}"
    )


def main():

    df = pd.read_csv("std_vs_error.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    plot_calibration(df, "energy_std", "energy_error", axes[0])
    plot_calibration(df, "force_max_std", "force_error", axes[1])
    plot_calibration(df, "force_rms_std", "force_error", axes[2])

    plt.tight_layout()

    plt.savefig("uncertainty_calibration.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()