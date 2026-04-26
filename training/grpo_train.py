import argparse
import asyncio
import ast
import os
import re
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.curriculum import DEFAULT_CURRICULUM, sample_difficulty
from training.rollout import run_episode


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ENV_URL = "http://127.0.0.1:8000"
VALID_ACTIONS = {
    "PROSPECT",
    "QUALIFY",
    "PRESENT",
    "HANDLE_OBJECTION",
    "OFFER_DEMO",
    "NEGOTIATE",
    "CLOSE",
    "FOLLOW_UP",
    "DISQUALIFY",
}
WORKFLOW_MAP = {
    1: ["QUALIFY", "PRESENT", "CLOSE"],
    2: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
    3: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
    4: [],
}


def _load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


async def curriculum_train(
    model,
    tokenizer,
    env_url: str,
    total_steps: int = 100,
    print_every: int = 10,
):
    """Curriculum rollout loop to benchmark env + policy behavior."""
    mean_reward = 0.0
    reward_history: list[float] = []
    run_log: list[dict] = []

    for step in range(total_steps):
        difficulty = sample_difficulty(DEFAULT_CURRICULUM, mean_reward)
        result = await run_episode(
            model=model,
            tokenizer=tokenizer,
            env_url=env_url,
            difficulty=difficulty,
        )

        reward_history.append(float(result["total_reward"]))
        mean_reward = float(np.mean(reward_history[-20:]))

        run_log.append(
            {
                "step": step,
                "difficulty": difficulty,
                "reward": float(result["total_reward"]),
                "violations": len(result["violations"]),
                "steps_completed": list(result["steps_completed"]),
            }
        )

        if step % print_every == 0:
            print(
                f"Step {step:04d} | Difficulty {difficulty} | "
                f"Reward {result['total_reward']:.3f} | Mean(20) {mean_reward:.3f} | "
                f"Violations {len(result['violations'])} | Steps {result['steps_completed']}"
            )

    return {
        "mean_reward": mean_reward,
        "reward_history": reward_history,
        "run_log": run_log,
    }


def _save_metrics(output_dir: str, metrics: dict):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rewards_path = output_path / "reward_history.txt"
    with rewards_path.open("w", encoding="utf-8") as f:
        for idx, reward in enumerate(metrics["reward_history"]):
            f.write(f"{idx}\t{reward:.6f}\n")
    print(f"Saved reward history to {rewards_path}")


def _extract_action_content(text: str) -> tuple[str, str]:
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    content_match = re.search(r"CONTENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    action_type = action_match.group(1).upper() if action_match else ""
    content = content_match.group(1).strip() if content_match else ""
    return action_type, content


def _extract_steps_completed(prompt_text: str) -> list[str]:
    match = re.search(r"Steps completed:\s*(\[.*?\])", prompt_text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
        if isinstance(parsed, list):
            return [str(v).upper() for v in parsed]
    except Exception:
        return []
    return []


def _extract_required_workflow(prompt_text: str) -> list[str]:
    match = re.search(r"Required workflow steps \(in order\):\s*(.+)", prompt_text)
    if not match:
        return []
    raw = match.group(1).strip()
    if raw.lower().startswith("dynamic"):
        return []
    return [part.strip().upper() for part in raw.split("->") if part.strip()]


def salespath_reward_func(prompts, completions, **kwargs):
    """
    Lightweight GRPO reward signal aligned with project rules.
    Uses format validity + basic workflow order constraints.
    """
    rewards: list[float] = []

    for prompt, completion in zip(prompts, completions):
        action_type, content = _extract_action_content(completion)
        steps_completed = _extract_steps_completed(prompt)
        required_workflow = _extract_required_workflow(prompt)

        reward = 0.0

        # Format + valid action (make this dense, not binary)
        has_action_prefix = "ACTION:" in completion.upper()
        has_content_prefix = "CONTENT:" in completion.upper()
        if has_action_prefix:
            reward += 0.05
        if has_content_prefix:
            reward += 0.05

        if action_type in VALID_ACTIONS:
            reward += 0.15
        else:
            rewards.append(-0.2)
            continue

        if content:
            reward += 0.1
        else:
            reward -= 0.1

        # Encourage concise responses so completions terminate before cap.
        content_len = len(content)
        if content_len > 220:
            reward -= 0.15
        elif content_len > 120:
            reward -= 0.05
        elif 12 <= content_len <= 120:
            reward += 0.05

        # Penalize rambling multi-paragraph completions.
        if completion.count("\n") > 4:
            reward -= 0.1

        # Positive signal for selecting the next expected workflow step.
        if required_workflow:
            next_idx = min(len(steps_completed), len(required_workflow) - 1)
            expected = required_workflow[next_idx]
            if action_type == expected:
                reward += 0.2

        # Rule hints
        if not steps_completed and action_type != "PROSPECT":
            reward -= 0.2  # R06
        if action_type == "PRESENT" and "QUALIFY" not in steps_completed:
            reward -= 0.2  # R01
        if action_type == "NEGOTIATE" and "OFFER_DEMO" not in steps_completed:
            reward -= 0.2  # R02
        if action_type == "CLOSE" and "OFFER_DEMO" not in steps_completed:
            reward -= 0.2  # R09

        # Keep rewards bounded for training stability.
        rewards.append(float(max(-1.0, min(1.0, reward))))

    return rewards


def _build_grpo_dataset_rows(num_rows: int = 128):
    rows = []
    prospect_snippets = [
        "We are evaluating options right now.",
        "Budget is tight this quarter.",
        "Can you explain implementation effort?",
        "Pricing seems high compared to alternatives.",
    ]

    for i in range(num_rows):
        difficulty = (i % 4) + 1
        workflow = WORKFLOW_MAP[difficulty]
        steps_completed = [] if i % 3 == 0 else workflow[: min(len(workflow), i % 2 + 1)]
        prompt = (
            "You are a B2B sales agent.\n\n"
            f"Required workflow steps (in order): {' -> '.join(workflow) if workflow else 'Dynamic'}\n"
            f"Current stage: {'START' if not steps_completed else steps_completed[-1]}\n"
            f"Steps completed: {steps_completed}\n"
            f"Turn: {(i % 8) + 1}/20\n"
            "Business rules: R01..R09 must be respected.\n"
            f"Prospect response: {prospect_snippets[i % len(prospect_snippets)]}\n\n"
            "Respond exactly with:\nACTION: <action>\nCONTENT: <message>"
        )
        rows.append({"prompt": prompt})
    return rows


def run_grpo(args):
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize TRL GRPO stack. On this machine, this is usually due to "
            "Windows blocking pyarrow dataset binaries in the local virtualenv. "
            "Use the provided Colab notebook (`training/colab_train.ipynb`) for GRPO runs, "
            "or fix local pyarrow/datasets installation first."
        ) from exc

    _, tokenizer = _load_model_and_tokenizer(args.model_name)
    rows = _build_grpo_dataset_rows(args.grpo_dataset_size)
    train_dataset = Dataset.from_list(rows)

    config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.grpo_steps,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        reward_funcs=salespath_reward_func,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(Path(args.output_dir) / "grpo_final"))
    print(f"Saved GRPO model to {Path(args.output_dir) / 'grpo_final'}")

    # --- Save reward history from trainer logs so plot_rewards.py works ---
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rewards_path = output_path / "reward_history.txt"

    log_rewards = []
    for entry in trainer.state.log_history:
        # TRL GRPO logs rewards under various key names depending on version
        for key in ("reward", "rewards", "mean_reward", "train/reward", "train/rewards"):
            if key in entry:
                log_rewards.append(float(entry[key]))
                break

    if log_rewards:
        with rewards_path.open("w") as f:
            for idx, r in enumerate(log_rewards):
                f.write(f"{idx}\t{r:.6f}\n")
        print(f"Saved reward history ({len(log_rewards)} entries) → {rewards_path}")
    else:
        # Fallback: write a placeholder so plot_rewards.py doesn't crash
        print("Warning: no reward entries found in trainer logs. Writing placeholder.")
        with rewards_path.open("w") as f:
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    f.write(f"{entry.get('step', 0)}\t0.0\n")

    if args.push_to_hub:
        trainer.push_to_hub(dataset_name="salespath_synthetic_grpo")
        print(f"Pushed trainer model to hub repo: {args.hub_repo}")


def parse_args():
    parser = argparse.ArgumentParser(description="SalesPath training entrypoint.")
    parser.add_argument("--mode", choices=["curriculum", "grpo"], default="curriculum")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--env-url", default=DEFAULT_ENV_URL)
    parser.add_argument("--steps", type=int, default=100, help="Curriculum rollout steps.")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output-dir", default="salespath_training_outputs")
    parser.add_argument("--hub-repo", default="Imsachin010/salespath-qwen25-7b")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--push-merged", action="store_true")

    # GRPO-specific knobs
    parser.add_argument("--grpo-steps", type=int, default=30)
    parser.add_argument("--grpo-dataset-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)

    return parser.parse_args()


async def _run_curriculum_mode(args):
    print(f"Loading model: {args.model_name}")
    model, tokenizer = _load_model_and_tokenizer(args.model_name)
    print(f"Starting curriculum loop against {args.env_url}")

    metrics = await curriculum_train(
        model=model,
        tokenizer=tokenizer,
        env_url=args.env_url,
        total_steps=args.steps,
        print_every=args.print_every,
    )
    print(f"Final mean reward (last 20): {metrics['mean_reward']:.4f}")
    _save_metrics(args.output_dir, metrics)

    if args.push_merged:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hasattr(model, "save_pretrained_merged"):
            merged_dir = Path(args.output_dir) / "salespath_trained_merged"
            model.save_pretrained_merged(
                str(merged_dir),
                tokenizer,
                save_method="merged_16bit",
            )
            print(f"Saved merged model to {merged_dir}")
            if hf_token and hasattr(model, "push_to_hub_merged"):
                model.push_to_hub_merged(
                    args.hub_repo,
                    tokenizer,
                    save_method="merged_16bit",
                    token=hf_token,
                )
                print(f"Pushed merged model to {args.hub_repo}")
        else:
            print(
                "Model does not support merged save APIs. "
                "Use an Unsloth merged-capable model to enable --push-merged."
            )


async def _main():
    args = parse_args()
    if args.mode == "curriculum":
        await _run_curriculum_mode(args)
        return

    print("Launching TRL GRPO mode...")
    run_grpo(args)


if __name__ == "__main__":
    asyncio.run(_main())

