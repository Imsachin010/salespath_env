# training/plot_rewards.py
# Run after grpo_train.py to generate reward graphs.
# Usage: python training/plot_rewards.py --input salespath_training_outputs/reward_history.txt

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np


def load_rewards(path: str) -> list[float]:
    rewards = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            rewards.append(float(parts[-1]))
    return rewards


def rolling_mean(data: list[float], window: int = 20) -> list[float]:
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(data[start : i + 1])))
    return result


def plot(rewards: list[float], output_path: str):
    steps = list(range(len(rewards)))
    smooth = rolling_mean(rewards, window=20)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("SalesPath — Training Reward Curve", fontsize=14, fontweight="bold")

    # Top: raw + smoothed reward
    ax = axes[0]
    ax.plot(steps, rewards, alpha=0.3, color="#5b9bd5", linewidth=0.8, label="Episode reward")
    ax.plot(steps, smooth, color="#e07b3c", linewidth=2.0, label="Rolling mean (20)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Total Reward")
    ax.set_xlabel("Episode")
    ax.legend(loc="upper left")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Bottom: histogram of rewards
    ax2 = axes[1]
    ax2.hist(rewards, bins=30, color="#5b9bd5", edgecolor="white", alpha=0.8)
    ax2.axvline(np.mean(rewards), color="#e07b3c", linewidth=2, label=f"Mean: {np.mean(rewards):.3f}")
    ax2.set_xlabel("Reward")
    ax2.set_ylabel("Count")
    ax2.set_title("Reward Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward graph → {output_path}")
    print(f"  Episodes:    {len(rewards)}")
    print(f"  Mean reward: {np.mean(rewards):.4f}")
    print(f"  Max reward:  {np.max(rewards):.4f}")
    print(f"  Min reward:  {np.min(rewards):.4f}")
    print(f"  Std reward:  {np.std(rewards):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot SalesPath reward history.")
    parser.add_argument(
        "--input",
        default="salespath_training_outputs/reward_history.txt",
        help="Path to reward_history.txt",
    )
    parser.add_argument(
        "--output",
        default="salespath_training_outputs/reward_graph.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run grpo_train.py first.")
        sys.exit(1)

    rewards = load_rewards(args.input)
    if not rewards:
        print("ERROR: No rewards found in file.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot(rewards, args.output)


if __name__ == "__main__":
    main()
