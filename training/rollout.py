# training/rollout.py
# Uses httpx (async HTTP) to call /reset and /step REST endpoints.
# This is robust for both local server and HF Space URL.

import sys
import os
import re
import httpx
import torch

# Ensure the project root is on sys.path regardless of how this is invoked
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from salespath_env.models import SalesPathObservation


SYSTEM_PROMPT = """You are a B2B sales agent. Your goal is to close deals by following a strict workflow.

Required workflow steps (in order): {workflow}

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

Respond EXACTLY in this format:
ACTION: <one of: PROSPECT, QUALIFY, PRESENT, HANDLE_OBJECTION, OFFER_DEMO, NEGOTIATE, CLOSE, FOLLOW_UP, DISQUALIFY>
CONTENT: <your message to the prospect>"""


def parse_action(text: str) -> tuple[str, str]:
    """Extract ACTION and CONTENT from model output."""
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    content_match = re.search(r"CONTENT:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    action_type = action_match.group(1).upper() if action_match else "QUALIFY"
    content = content_match.group(1).strip() if content_match else "Tell me more about your needs."
    return action_type, content


def build_prompt(obs: SalesPathObservation, workflow: list[str], tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(
            workflow=" -> ".join(workflow) if workflow else "Dynamic — determine best path"
        )},
        {"role": "user", "content": (
            f"Prospect response: {obs.prospect_response}\n"
            f"Current stage: {obs.workflow_stage}\n"
            f"Steps completed: {obs.steps_completed}\n"
            f"Turn: {obs.turn_number}/20\n"
            f"Violations so far: {obs.constraints_violated}\n\n"
            "What is your next action?"
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_obs(data: dict) -> SalesPathObservation:
    """Parse observation — handles nested 'observation' key from server."""
    obs_data = data.get("observation", data)
    # Drop unknown keys that Pydantic would reject
    known = SalesPathObservation.model_fields.keys()
    obs_data = {k: v for k, v in obs_data.items() if k in known}
    return SalesPathObservation(**obs_data)


async def run_episode(
    model,
    tokenizer,
    env_url: str,
    difficulty: int = 1,
    message_timeout_s: float = 300.0,
) -> dict:
    """Run one full episode via REST. Returns trajectory + rewards."""
    DIFFICULTY_WORKFLOW = {
        1: ["QUALIFY", "PRESENT", "CLOSE"],
        2: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
        3: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO",
            "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
        4: [],
    }
    workflow = DIFFICULTY_WORKFLOW[difficulty]

    async with httpx.AsyncClient(base_url=env_url, timeout=message_timeout_s) as client:

        # --- Reset ---
        reset_resp = await client.post("/reset", json={"difficulty": difficulty})
        reset_resp.raise_for_status()
        obs = _parse_obs(reset_resp.json())

        trajectory = []
        total_reward = 0.0

        while not obs.done:
            # --- Model inference ---
            prompt = build_prompt(obs, workflow, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            action_type, content = parse_action(generated)

            # --- Step via REST ---
            step_resp = await client.post(
                "/step",
                json={"action": {"action_type": action_type, "content": content, "target": ""}},
            )
            step_resp.raise_for_status()
            obs = _parse_obs(step_resp.json())

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