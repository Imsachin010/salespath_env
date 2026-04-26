# training/test_rollout.py
# Works both as: python training/test_rollout.py
# And as:        python -m training.test_rollout

import sys
import os
import asyncio

# Add BOTH project root AND training/ dir so imports work in all contexts
_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.abspath(os.path.dirname(__file__))
for _p in [_ROOT, _TRAINING]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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