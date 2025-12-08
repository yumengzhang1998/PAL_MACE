from matplotlib import scale
import pandas as pd
import numpy as np
import re

def read_rmse(path):
    with open(path) as f:
        log_data = f.read()
    energy_rmse = re.search(r"Energy RMSE:\s+([0-9.]+)", log_data)
    energy_r2 = re.search(r"Energy R2:\s+([-0-9.]+)", log_data)
    energy_mae = re.search(r"Energy MAE:\s+([0-9.]+)", log_data)
    forces_rmse = re.search(r"Forces RMSE:\s+([0-9.]+)", log_data)
    forces_r2 = re.search(r"Forces R2:\s+([-0-9.]+)", log_data)

    return (
        float(energy_rmse.group(1)),
        float(energy_r2.group(1)),
        float(energy_mae.group(1)),
        float(forces_rmse.group(1)),
        float(forces_r2.group(1))
    )



def save_rmse(prefixes, model, num_samples):
    results = []

    for p in prefixes:
        e_rmse, e_r2, e_mae, f_rmse, f_r2 = [], [], [], [], []
        for i in range(num_samples):
            path = f'{p}_logs/sample_{i}/logs/{p}_run-123.log'
            try:
                energy_rmse, energy_r2, energy_mae, forces_rmse, forces_r2 = read_rmse(path)
                e_rmse.append(energy_rmse)
                e_r2.append(energy_r2)
                e_mae.append(energy_mae)
                f_rmse.append(forces_rmse)
                f_r2.append(forces_r2)
            except Exception as e:
                print(f"⚠️ Skipped {path} due to error: {e}")

        # Aggregate means
        results.append({
            "prefix": p,
            "Energy RMSE": np.mean(e_rmse),
            "Energy R2": np.mean(e_r2),
            "Energy MAE": np.mean(e_mae),
            "Forces RMSE": np.mean(f_rmse),
            "Forces R2": np.mean(f_r2)
        })

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(f'./summary_rmse.csv', index=False)
    print("✅ Saved summary to summary_rmse.csv")



prefixes = ['bi0']
num_samples = 5
m = 'full_data_charge_embed'
save_rmse(prefixes, m, num_samples)