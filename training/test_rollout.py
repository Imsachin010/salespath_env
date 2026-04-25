# training/test_rollout.py

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rollout import run_episode
except ImportError:
    from training.rollout import run_episode


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Local server (already running via uvicorn) — more reliable than HF Space WS
ENV_URL = "http://127.0.0.1:8000"


async def main():
    print("Loading small local model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype="auto",
        device_map="auto",
    )

    print("Running single episode...")

    result = await run_episode(
        model=model,
        tokenizer=tokenizer,
        env_url=ENV_URL,
        difficulty=1,
        message_timeout_s=300.0,   # allow up to 5 min per step (CPU inference is slow)
    )

    print("\n========== RESULT ==========")
    print(
        f"Total Reward: {result['total_reward']:.4f}"
    )
    print(
        f"Violations: {result['violations']}"
    )
    print(
        f"Steps Completed: {result['steps_completed']}"
    )

    if result["trajectory"]:
        print("\n=== First Generation ===")
        print(
            result["trajectory"][0]["generated"]
        )


if __name__ == "__main__":
    asyncio.run(main())