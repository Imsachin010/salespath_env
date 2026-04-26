# training/test_rollout.py
# Run from project root: .\.spa\Scripts\python.exe training\test_rollout.py

import sys
import os
import asyncio

# Ensure project root is on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from rollout import run_episode


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ENV_URL = "http://127.0.0.1:8000"


async def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Model on: {next(model.parameters()).device}")
    print("Running single episode (difficulty=1)...\n")

    result = await run_episode(
        model=model,
        tokenizer=tokenizer,
        env_url=ENV_URL,
        difficulty=1,
        message_timeout_s=300.0,
    )

    print("\n========== RESULT ==========")
    print(f"Total Reward:     {result['total_reward']:.3f}")
    print(f"Violations:       {result['violations']}")
    print(f"Steps Completed:  {result['steps_completed']}")
    print(f"Difficulty:       {result['difficulty']}")
    print("=============================\n")

    if result["trajectory"]:
        first = result["trajectory"][0]
        print("=== First Generation ===")
        print(f"ACTION:  {first['action_type']}")
        print(f"CONTENT: {first['generated'][:200]}")
        print(f"REWARD:  {first['reward']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())