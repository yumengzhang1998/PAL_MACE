import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


df = pd.read_csv("std_vs_error.csv")

pairs = [
    ("energy_std", "energy_error"),
    ("force_max_std", "force_error"),
    ("force_rms_std", "force_error"),
]

for u, e in pairs:

    corr, p = spearmanr(df[u], df[e])

    print(f"{u} vs {e}")
    print("Spearman:", corr)
    print("p-value:", p)
    print()

    plt.figure()

    plt.scatter(df[u], df[e], alpha=0.6)

    plt.xlabel(u)
    plt.ylabel(e)

    plt.title(f"{u} vs {e}\nSpearman={corr:.3f}")

    plt.tight_layout()

    plt.savefig(f"{u}_vs_{e}.png", dpi=300)

plt.show()