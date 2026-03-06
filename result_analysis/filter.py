import re
import pandas as pd
import matplotlib.pyplot as plt

def analyze_filter_log(path):
    kept_list = []
    total_list = []

    pattern = r"After filtering:\s+(\d+)\s+kept out of\s+(\d+)"

    with open(path, "r") as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                kept = int(m.group(1))
                total = int(m.group(2))

                kept_list.append(kept)
                total_list.append(total)

    df = pd.DataFrame({
        "iteration": range(1, len(kept_list) + 1),
        "kept": kept_list,
        "total": total_list
    })

    return df


def plot_kept_trend(df, save_name="kept_trend.png"):
    plt.figure(figsize=(10,6))

    # Plot kept and total
    plt.plot(df["iteration"], df["kept"], 
             label="Kept", marker="o", linewidth=2)
    plt.plot(df["iteration"], df["total"], 
             label="Total", marker="o", linewidth=2, alpha=0.6)

    # Fill between to show proportion
    plt.fill_between(df["iteration"], df["kept"], alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.title("Trend of kept vs total across filtering iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved trend figure to {save_name}")


# ========================
# Usage
# ========================

path = "../script/syn_3872972.out"
df = analyze_filter_log(path)


plot_kept_trend(df)
