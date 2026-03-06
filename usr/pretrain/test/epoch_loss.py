import re
import sys
sys.path.append("../")
from plot import plot_loss_epoch_together
def parse_log_file(log_file_path):
    # Lists to store extracted values
    epochs = []
    losses = []
    mae_e_values = []
    mae_f_values = []

    # Regular expression pattern to match epoch lines
    pattern = re.compile(r"Epoch (\d+): head: default, loss=([\d.]+), MAE_E=\s*([\d.]+) meV, MAE_F=\s*([\d.]+) meV / A")

    with open(log_file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                mae_e_values.append(float(match.group(3)))
                mae_f_values.append(float(match.group(4)))

    return epochs, losses, mae_e_values, mae_f_values

# Example usage
prefix = 'bi4-2'
log_file_path = f"../{prefix}_logs_e0_after/sample_0/logs/"  # Update with your actual log file path
log_name = "bi4-2_run-123.log"
epochs, losses, mae_e_values, mae_f_values = parse_log_file(log_file_path+ log_name)

# Print extracted data types to verify
print(f"Epochs type: {type(epochs[0])}, Example: {epochs[0]}")
print(f"Losses type: {type(losses[0])}, Example: {losses[0]}")
print(f"MAE_E type: {type(mae_e_values[0])}, Example: {mae_e_values[0]}")
print(f"MAE_F type: {type(mae_f_values[0])}, Example: {mae_f_values[0]}")
plot_loss_epoch_together(losses, mae_e_values, mae_f_values, log_file_path, 'charge embedding after node embedding with dft E0')

