# training/rollout.py

import re
import torch

from salespath_env.client import SalesPathEnv
from salespath_env.models import SalesPathObservation


SYSTEM_PROMPT = """
You are a B2B sales agent.

Your goal is to close deals by following a strict workflow.

Required workflow steps (in order):
{workflow}

Business rules — NEVER violate these:

- R01: Must QUALIFY before PRESENT
- R02: Must OFFER_DEMO before NEGOTIATE
- R03: Budget must be known before NEGOTIATE
- R04: Discount only after 2 objections handled
- R05: Cannot repeat same action twice in a row
- R06: First action must always be PROSPECT
- R07: FOLLOW_UP only after prospect goes silent
- R08: DISQUALIFY only if prospect is genuinely unqualified
- R09: Must OFFER_DEMO before CLOSE (difficulty 2+)

You must respond EXACTLY in this format:

ACTION: <one valid action>
CONTENT: <your message>
"""


def parse_action(text: str) -> tuple[str, str]:
    """
    Extract ACTION and CONTENT from model output.
    Fallback = QUALIFY if parsing fails.
    """
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    content_match = re.search(r"CONTENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)

    action_type = action_match.group(1).upper() if action_match else "QUALIFY"
    content = content_match.group(1).strip() if content_match else "Tell me more about your current process."

    return action_type, content


def build_prompt(obs: SalesPathObservation, workflow: list[str], tokenizer) -> str:
    """Build model prompt from environment observation."""
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(workflow=" -> ".join(workflow)),
        },
        {
            "role": "user",
            "content": (
                f"Prospect response: {obs.prospect_response}\n"
                f"Current stage: {obs.workflow_stage}\n"
                f"Steps completed: {obs.steps_completed}\n"
                f"Turn: {obs.turn_number}/20\n"
                f"Violations so far: {obs.constraints_violated}\n\n"
                "What is your next action?"
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


async def run_episode(
    model,
    tokenizer,
    env_url: str,
    difficulty: int = 1,
    message_timeout_s: float = 300.0,
) -> dict:
    """
    Run one full episode using the stateful OpenEnv client.
    Returns trajectory + rewards.
    """
    DIFFICULTY_WORKFLOW = {
        1: ["QUALIFY", "PRESENT", "CLOSE"],
        2: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
        3: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
        4: [],
    }

    workflow = DIFFICULTY_WORKFLOW[difficulty]

    async with SalesPathEnv(base_url=env_url) as env:
        obs = await env.reset(difficulty=difficulty)
        trajectory = []
        total_reward = 0.0

        while not obs.done:
            # --- Model inference (CPU/GPU — no network) ---
            prompt = build_prompt(obs, workflow, tokenizer)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            action_type, content = parse_action(generated)

            # --- Stateful step via OpenEnv client ---
            obs = await env.step(
                action_type=action_type,
                content=content,
                target="",
            )

            trajectory.append({
                "prompt": prompt,
                "generated": generated,
                "action_type": action_type,
                "reward": obs.reward,
                "components": obs.reward_components,
                "done": obs.done,
            })

            total_reward += obs.reward

    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "steps_completed": obs.steps_completed,
        "violations": obs.constraints_violated,
        "difficulty": difficulty,
    }