"""
SalesPath — Evaluate Baseline vs Trained Model

Runs episodes at each difficulty level with both the base model
and the trained (GRPO) model, then compares performance.

Usage:
    python training/eval_baseline_vs_trained.py \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --trained ./salespath_out/grpo_final \
        --env-url http://127.0.0.1:8000 \
        --episodes-per-level 4 \
        --output ./salespath_out/eval_results.json
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.rollout import run_episode

DIFFICULTIES = [1, 2, 3, 4]


async def eval_model(
    model,
    tokenizer,
    env_url: str,
    episodes_per_level: int = 4,
    label: str = "model",
) -> dict:
    """Evaluate a model across all difficulty levels."""
    results = {
        "label": label,
        "per_difficulty": {},
        "overall": {},
    }

    all_rewards = []
    all_violations = []
    all_closes = []
    all_lengths = []

    for difficulty in DIFFICULTIES:
        diff_rewards = []
        diff_violations = []
        diff_closes = []
        diff_lengths = []

        print(f"  Difficulty {difficulty}...")
        for ep in range(episodes_per_level):
            result = await run_episode(
                model=model,
                tokenizer=tokenizer,
                env_url=env_url,
                difficulty=difficulty,
                message_timeout_s=120.0,
            )

            trajectory = result["trajectory"]
            reward = result["total_reward"]
            violations = result["violations"]
            steps = result["steps_completed"]
            length = len(trajectory)

            # Did we close successfully?
            last_action = trajectory[-1]["action_type"] if trajectory else ""
            last_traj = trajectory[-1] if trajectory else {}
            components = last_traj.get("components", {})
            r_outcome = components.get("r_outcome", 0.0)
            closed = last_action == "CLOSE" and r_outcome > 0

            diff_rewards.append(reward)
            diff_violations.append(len(violations))
            diff_closes.append(1 if closed else 0)
            diff_lengths.append(length)

            all_rewards.append(reward)
            all_violations.append(len(violations))
            all_closes.append(1 if closed else 0)
            all_lengths.append(length)

        results["per_difficulty"][difficulty] = {
            "mean_reward": sum(diff_rewards) / len(diff_rewards) if diff_rewards else 0,
            "mean_violations": sum(diff_violations) / len(diff_violations) if diff_violations else 0,
            "close_rate": sum(diff_closes) / len(diff_closes) if diff_closes else 0,
            "mean_episode_length": sum(diff_lengths) / len(diff_lengths) if diff_lengths else 0,
            "num_episodes": len(diff_rewards),
        }

    results["overall"] = {
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        "mean_violations": sum(all_violations) / len(all_violations) if all_violations else 0,
        "close_rate": sum(all_closes) / len(all_closes) if all_closes else 0,
        "mean_episode_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
        "num_episodes": len(all_rewards),
    }

    return results


def load_model(model_name_or_path: str):
    """Load model, detecting if it's a PEFT adapter."""
    print(f"Loading model: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a PEFT adapter directory
    adapter_path = Path(model_name_or_path)
    is_adapter = (adapter_path / "adapter_config.json").exists()

    if is_adapter:
        print("  Detected PEFT adapter — loading base model + adapter...")
        from peft import PeftModel

        # Find the base model name from adapter config
        import json as _json
        with open(adapter_path / "adapter_config.json") as f:
            adapter_cfg = _json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B-Instruct")
        print(f"  Base model: {base_model_name}")

        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if bf16_supported else torch.float32,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        # Merge adapter for faster inference
        model = model.merge_and_unload()
        print("  Adapter loaded and merged ✅")
    else:
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if bf16_supported else torch.float32,
            device_map="auto",
        )

    model.eval()
    print(f"  Model on: {next(model.parameters()).device}")
    return model, tokenizer


async def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs trained model")
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--trained", default="./salespath_out/grpo_final")
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--episodes-per-level", type=int, default=4)
    parser.add_argument("--output", default="./salespath_out/eval_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("SalesPath — Model Evaluation")
    print("=" * 60)
    print(f"Base model:    {args.base}")
    print(f"Trained model: {args.trained}")
    print(f"Episodes/level: {args.episodes_per_level}")
    print()

    # Evaluate base model
    print("Loading base model...")
    base_model, base_tokenizer = load_model(args.base)
    print("\nEvaluating base model...")
    base_results = await eval_model(
        base_model, base_tokenizer, args.env_url,
        episodes_per_level=args.episodes_per_level,
        label="baseline",
    )

    # Clean up
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate trained model
    print("\nLoading trained model...")
    trained_model, trained_tokenizer = load_model(args.trained)
    print("\nEvaluating trained model...")
    trained_results = await eval_model(
        trained_model, trained_tokenizer, args.env_url,
        episodes_per_level=args.episodes_per_level,
        label="trained",
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    for model_results in [base_results, trained_results]:
        label = model_results["label"]
        overall = model_results["overall"]
        print(f"\n--- {label.upper()} ---")
        print(f"  Mean reward:       {overall['mean_reward']:.4f}")
        print(f"  Mean violations:   {overall['mean_violations']:.2f}")
        print(f"  Close rate:        {overall['close_rate']:.2%}")
        print(f"  Mean ep. length:   {overall['mean_episode_length']:.1f}")

        for diff, metrics in model_results["per_difficulty"].items():
            print(f"  Difficulty {diff}: reward={metrics['mean_reward']:.3f}, "
                  f"violations={metrics['mean_violations']:.1f}, "
                  f"close={metrics['close_rate']:.0%}")

    # Save results
    output = {
        "base": base_results,
        "trained": trained_results,
        "comparison": {
            "reward_delta": trained_results["overall"]["mean_reward"] - base_results["overall"]["mean_reward"],
            "violation_reduction": base_results["overall"]["mean_violations"] - trained_results["overall"]["mean_violations"],
            "close_rate_improvement": trained_results["overall"]["close_rate"] - base_results["overall"]["close_rate"],
        },
        "config": {
            "base_model": args.base,
            "trained_model": args.trained,
            "episodes_per_level": args.episodes_per_level,
            "difficulties": DIFFICULTIES,
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print comparison summary
    print("\n=== KEY METRICS ===")
    c = output["comparison"]
    print(f"  Reward delta:        {c['reward_delta']:+.4f}")
    print(f"  Violation reduction: {c['violation_reduction']:+.2f}")
    print(f"  Close rate change:   {c['close_rate_improvement']:+.2%}")


if __name__ == "__main__":
    asyncio.run(main())
