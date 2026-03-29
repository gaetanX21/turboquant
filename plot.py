from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from fire import Fire
from turboquant import T, TurboQuant

sns.set_theme(style="whitegrid", palette="muted")

FIG_DIR = Path(__name__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def error_statistics(d: int, b: int, N: int) -> tuple[T, T]:
    """Compute error statistics for TurboQuant quantization algorithm."""
    tq = TurboQuant(d=d, b=b)
    X = torch.randn(N, d)
    # normalized vectors for normalized error statistics
    X = X / torch.linalg.vector_norm(X, dim=1, keepdim=True)
    X_hat = tq.dequantize(*tq.quantize(X))
    coordinate_diff = (X - X_hat).reshape(-1)
    vector_dist = torch.linalg.norm(X - X_hat, dim=1)
    return coordinate_diff, vector_dist


def plot_error_statistics(d: int, b: int, N: int) -> None:
    coordinate_diff, vector_dist = error_statistics(d, b, N)

    # Convert to numpy arrays to compute stats and plot
    coord_np = coordinate_diff.numpy()
    vec_np = vector_dist.numpy()

    # Compute mean and standard deviation
    mu_coord, sigma_coord = coord_np.mean(), coord_np.std()
    mu_vec, sigma_vec = vec_np.mean(), vec_np.std()

    # Create a 1x2 grid (side-by-side) for better widescreen viewing
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"TurboQuant Error Distributions (Dimension={d}, Bits={b})",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    # Define bounding box properties
    bbox_props = dict(
        boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
    )

    # Plot 1: Coordinate Differences
    sns.histplot(
        coord_np,
        bins=100,
        ax=ax0,
        color="royalblue",
        stat="density",
        alpha=0.7,
    )
    ax0.set_title("Coordinate Difference Distribution", fontsize=14)
    ax0.set_xlabel("Coordinate Error $(X_i - \\hat{X}_i)$", fontsize=12)
    ax0.set_ylabel("Density", fontsize=12)

    # Add text box for Plot 1
    ax0.text(
        0.95,
        0.95,
        f"$\\mu = {mu_coord:.4f}$\n$\\sigma = {sigma_coord:.4f}$",
        transform=ax0.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )

    # Plot 2: Vector Distances
    sns.histplot(
        vec_np,
        bins=100,
        ax=ax1,
        color="darkorange",
        stat="density",
        alpha=0.7,
    )
    ax1.set_title("Vector Distance Distribution", fontsize=14)
    ax1.set_xlabel("$||X - \\hat{X}||_2$", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)

    # Add text box for Plot 2
    ax1.text(
        0.95,
        0.95,
        f"$\\mu = {mu_vec:.4f}$\n$\\sigma = {sigma_vec:.4f}$",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )

    # Adjust layout to prevent clipping
    plt.tight_layout()
    # Save fig
    plt.savefig(FIG_DIR / f"{b=}_{d=}_{N=}.png")


def main(b_list: list[int] = [1, 2, 3, 4], d: int = 1024, N: int = 10_000) -> None:
    for b in b_list:
        print(f"Plotting error statistics for {b=}, {d=}, {N=}")
        plot_error_statistics(d, b, N)
    print(f"Plots saved in {FIG_DIR=}")


if __name__ == "__main__":
    Fire(main)
    """
    python plot.py 
    """
