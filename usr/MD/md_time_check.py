import re
import glob
import numpy as np
import matplotlib.pyplot as plt

out_files = glob.glob("bisyn_2nstraj_*K_*.out")

pattern_out = re.compile(
    r"step\s+(\d+)\s+\|\s+mace\s+([\d\.]+)s\s+\|\s+openmm\+sync\s+([\d\.]+)s"
)

pattern_err = re.compile(
    r"\|\s*(\d+)/\d+.*?,\s*([\d\.]+)it/s"
)

jobid_pattern = re.compile(r"bisyn_2nstraj_\d+K_(\d+)")

for out_file in out_files:

    jobid = jobid_pattern.search(out_file).group(1)
    err_file = out_file.replace(".out", ".err")

    # ---- read it/s from ERR ----
    its_dict = {}

    with open(err_file) as f:
        for line in f:
            m = pattern_err.search(line)
            if m:
                step = int(m.group(1))
                its = float(m.group(2))
                its_dict[step] = its

    # ---- read timings from OUT ----
    steps = []
    mace_times = []
    openmm_times = []
    total_step_times = []

    with open(out_file) as f:
        for line in f:
            m = pattern_out.search(line)
            if m:
                step = int(m.group(1))
                mace = float(m.group(2))
                openmm = float(m.group(3))

                steps.append(step)
                mace_times.append(mace)
                openmm_times.append(openmm)

                if step in its_dict:
                    total_step_times.append(1 / its_dict[step])
                else:
                    total_step_times.append(np.nan)
    print(total_step_times  )
    # ---- plot ----
    plt.figure()

    plt.plot(steps, mace_times, label="MACE time")
    plt.plot(steps, openmm_times, label="OpenMM+sync time")
    plt.plot(steps, total_step_times, label="Total step time (1/it/s)")

    plt.xlabel("Step")
    plt.ylabel("Time per step (s)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{jobid}_md_time.png")
    plt.close()

    print(f"Saved {jobid}_md_time.png")