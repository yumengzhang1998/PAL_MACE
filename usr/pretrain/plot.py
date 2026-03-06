# to plot true vs predicted energies and forces

from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_predicted(truth: List, prediction: List, save_path,Lable = "Energy"):
    """
    Plot true vs predicted energies and forces, 
    """
    truth = np.array(truth)
    prediction = np.array(prediction)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.scatter(truth,  prediction, alpha=0.5)
    min_val = min(truth.min(), prediction.min())
    max_val = max(truth.max(), prediction.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
    ax.set_xlabel(f"True {Lable} (eV)")
    ax.set_ylabel(f"Predicted {Lable} (eV)")
    ax.set_title(f"True vs Predicted {Lable}")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{Lable}_pred.png")
    plt.close(fig)


def plot_distribution(property: List, save_path, Lable = "Energy"):
    """
    Plot distribution of a property
    """
    property = np.array(property)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.hist(property, bins=50, alpha=0.5)
    ax.set_xlabel(f"{Lable} (eV)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {Lable}")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{Lable}_distribution.png")
    plt.close(fig)

def plot_loss_epoch(loss: List, save_path, Lable = "Loss"):
    """
    Plot loss vs epoch
    """
    loss = np.array(loss)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    ax.plot(range(len(loss)), loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{Lable}")
    ax.set_title(f"{Lable} vs Epoch")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{Lable}_epoch.png")
    plt.close(fig)

def plot_loss_epoch_together(loss: List, e_loss: List, f_loss: List,save_path, title: str,Lable = "MACE with charge"):
    """
    Plot loss vs epoch
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    epochs = list(range(1, len(loss) + 1))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Plot Loss

    axes[0].plot(epochs, loss, label='Loss', color='blue')
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Energy Loss
    axes[1].plot(epochs, e_loss, label='Energy Loss', color='red')
    axes[1].set_ylabel("Energy Loss")
    axes[1].legend()
    axes[1].grid(True)

    # Plot Force Loss
    axes[2].plot(epochs, f_loss, label='Force Loss', color='green')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Force Loss")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{Lable}_epoch.png")
    plt.close(fig)