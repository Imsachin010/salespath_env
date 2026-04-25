import argparse
import asyncio

from salespath_env.client import SalesPathEnv


async def run_debug(env_url: str, difficulty: int):
    actions = [
        ("PRESENT", "pitch too early"),
        ("PRESENT", "repeat pitch"),
        ("PRESENT", "repeat pitch again"),
    ]

    async with SalesPathEnv(base_url=env_url) as env:
        obs = await env.reset(difficulty=difficulty)
        print("RESET")
        print(f"  turn={obs.turn_number} done={obs.done} reward={obs.reward}")
        print(f"  response={obs.prospect_response}")

        for idx, (action_type, content) in enumerate(actions, start=1):
            obs = await env.step(action_type=action_type, content=content, target="")
            print(f"\nSTEP {idx} action={action_type}")
            print(f"  turn={obs.turn_number} done={obs.done} reward={obs.reward}")
            print(f"  violations={obs.constraints_violated}")
            print(f"  new_violations={obs.info.get('new_violations')}")
            print(f"  components={obs.reward_components}")
            if obs.done:
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Debug stateful episode transitions.")
    parser.add_argument("--env-url", default="http://127.0.0.1:8000")
    parser.add_argument("--difficulty", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_debug(args.env_url, args.difficulty))
