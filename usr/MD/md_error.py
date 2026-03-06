import re
import glob
import numpy as np
import matplotlib.pyplot as plt

# find all matching files
files = glob.glob("bisyn_2nstraj_*K_*.err")

pattern = re.compile(r'\|\s*(\d+)/\d+.*?,\s*([\d\.]+)it/s')
jobid_pattern = re.compile(r'bisyn_2nstraj_\d+K_(\d+)\.err')

for filename in files:
    n_values = []
    its_values = []

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                n = int(match.group(1))
                its = float(match.group(2))
                n_values.append(n)
                its_values.append(its)

    print(f"{filename}: Parsed {len(n_values)} data points")

    if len(n_values) == 0:
        continue

    # extract jobid
    jobid_match = jobid_pattern.search(filename)
    if jobid_match:
        jobid = jobid_match.group(1)
    else:
        jobid = "unknown"

    # ---- Plot ----
    plt.figure()
    plt.plot(n_values, its_values)

    # trend line
    z = np.polyfit(n_values, its_values, 1)
    p = np.poly1d(z)
    plt.plot(n_values, p(n_values), "r--",
             label=f"Trend line: y={z[0]:.2f}x + {z[1]:.2f}")

    plt.xlabel("Loop number (n)")
    plt.ylabel("Iterations per second (it/s)")
    plt.legend()
    plt.tight_layout()

    # save plot
    plt.savefig(f"{jobid}_md_error.png")
    plt.close()