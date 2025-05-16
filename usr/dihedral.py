from sys import prefix
import pandas as pd
import json
import matplotlib.pyplot as plt
import argparse


def retraining_trend(number, json_file_path):
    # Load the JSON data
    with open(f'{json_file_path}/retrain_history_{number}.json', 'r') as f:
        data = json.load(f)

    # Access the "MSE_train" and "MSE_val" lists
    mse_train = data.get("MSE_train", [])
    mse_val = data.get("MSE_val", [])
    print(len(mse_train), len(mse_val))
    plt.figure(figsize=(12, 6))  # Create a new figure with a specified size
    plt.subplot(1, 2, 1)  # Create a 1x2 grid, and select the first plot
    plt.scatter(range(len(mse_train)), mse_train, color='blue', label='Train MSE')  # Plot training MSE
    plt.title('Training MSE')
    plt.xlabel('Retraining Instance')
    plt.ylabel('MSE')
    plt.xticks(range(0, len(mse_train), 2))  # Ensure x-axis has only integer values
    plt.grid(True)

    # Create scatter plot for MSE_val
    plt.subplot(1, 2, 2)  # Select the second plot
    plt.scatter(range(len(mse_val)), mse_val, color='red', label='Validation MSE')  # Plot validation MSE
    plt.title('Validation MSE')
    plt.xlabel('Retraining Instance')
    plt.ylabel('MSE')
    plt.xticks(range(0, len(mse_train), 2))  # Ensure x-axis has only integer values
    plt.grid(True)

    # Show the plot
    plt.tight_layout()  # Adjust subplot parameters to give some padding
    plt.savefig(f"{json_file_path}/mse_{number}.png")  # Save the plot as an image
    plt.show()  # Display the scatter plots
    # Path to your JSON file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate batch trajectories.")
    parser.add_argument("--element", type=str, required=True, help="Element symbol (e.g., 'bi')")
    parser.add_argument("--charge", type=int, required=True, help="Charge of the system (e.g., -2)")
    parser.add_argument("--num_atom", type=int, required=True, help="Number of atoms (e.g., 4)")
    parser.add_argument("--model_number", type=int, required=True, help="Model number (e.g., 25)")
    parser.add_argument("--synthesis", type=str, required=True, help="if the data is from synthesis")
    args = parser.parse_args()
    if args.synthesis == "True":
        prefix = f"{args.element}{args.num_atom}{args.charge}_samples"
    else:
        prefix = f"{args.element}{args.num_atom}{args.charge}"
    number = args.model_number
    json_file_path = f"../results/{prefix}/"
    retraining_trend(number, json_file_path)

