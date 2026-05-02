import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot a saved GSDiff loss_curve.npy file.")
    parser.add_argument("loss_curve", type=Path, help="Path to loss_curve.npy")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output image path. Defaults to loss_curve.jpg next to the .npy file.",
    )
    args = parser.parse_args()

    loss_curve = np.load(args.loss_curve)
    if loss_curve.ndim == 1:
        loss_curve = loss_curve[:, None]

    output = args.output or args.loss_curve.with_suffix(".jpg")
    output.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(1, len(loss_curve) + 1)
    plt.figure(figsize=(10, 6))
    for column in range(loss_curve.shape[1]):
        plt.plot(steps, loss_curve[:, column], label=f"loss {column}")

    plt.xlabel("Logged training step")
    plt.ylabel("Loss")
    plt.title(args.loss_curve.parent.name)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
