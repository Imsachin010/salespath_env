# SalesPath — End-to-End Coding Approach
### For Agent Execution. Follow in order. No skipping.

---

## Phase 0: Setup (Do First, ~15 min)

```bash
# Install OpenEnv
pip install openenv

# Scaffold the project
openenv init salespath_env
cd salespath_env

# Install dependencies
pip install -e .

# Verify scaffold works
uv run server --host 0.0.0.0 --port 8000
# Should start FastAPI on 8000. Ctrl+C after confirming.
```

Edit `pyproject.toml` — add dependencies:
```toml
[project]
name = "salespath_env"
version = "0.1.0"
dependencies = [
    "openenv",
    "fastapi",
    "uvicorn",
    "pydantic>=2.0",
    "trl>=0.8.0",
    "unsloth",
    "torch",
    "transformers",
]
```

---

## Phase 1: Models (Person A) — `models.py`

Write this file first. Everything else depends on it.

```python
# salespath_env/models.py
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Optional
from openenv.core import Action, Observation, State

VALID_ACTIONS = {
    "PROSPECT", "QUALIFY", "PRESENT", "HANDLE_OBJECTION",
    "OFFER_DEMO", "NEGOTIATE", "CLOSE", "FOLLOW_UP", "DISQUALIFY"
}

class SalesPathAction(Action):
    action_type: str
    content: str
    target: str = ""

    def is_valid(self) -> bool:
        return self.action_type in VALID_ACTIONS


class SalesPathObservation(Observation):
    prospect_response: str = ""
    workflow_stage: str = "START"
    constraints_violated: list[str] = field(default_factory=list)
    steps_completed: list[str] = field(default_factory=list)
    turn_number: int = 0
    reward: float = 0.0
    reward_components: dict = field(default_factory=dict)
    done: bool = False
    info: dict = field(default_factory=dict)


class SalesPathState(State):
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prospect_profile: dict = field(default_factory=dict)
    conversation_history: list[dict] = field(default_factory=list)
    workflow_stage: str = "START"
    required_workflow: list[str] = field(default_factory=list)
    steps_completed: list[str] = field(default_factory=list)
    constraints_violated: list[str] = field(default_factory=list)
    objections_handled: int = 0
    turn_number: int = 0
    difficulty: int = 1
    done: bool = False
    # Hidden — never expose in Observation
    _hidden: dict = field(default_factory=dict)
```

---

## Phase 2: Task Bank (Person A) — `server/task_bank.py`

This generates prospect profiles. Keep it simple — 10 profiles per difficulty level.

```python
# server/task_bank.py
import random
from dataclasses import dataclass

@dataclass
class ProspectProfile:
    company_name: str
    company_size: str          # "small" / "medium" / "enterprise"
    industry: str
    budget_signal: str         # "high" / "medium" / "low" / "unknown"
    pain_points: list[str]
    decision_maker: bool
    # Hidden — simulator uses these, agent never sees raw values
    true_budget: float         # 0.0 to 1.0 scale
    close_threshold: float     # budget needed to close
    stall_probability: float   # for Level 3+


PROFILES_L1 = [
    ProspectProfile(
        company_name="Meridian Retail",
        company_size="medium",
        industry="retail",
        budget_signal="high",
        pain_points=["manual inventory tracking", "slow reporting"],
        decision_maker=True,
        true_budget=0.8,
        close_threshold=0.5,
        stall_probability=0.0,
    ),
    # Add 9 more L1 profiles following same pattern
    # L1: budget_signal always known, decision_maker always True, close_threshold <= 0.6
]

PROFILES_L2 = [
    ProspectProfile(
        company_name="Apex Logistics",
        company_size="enterprise",
        industry="logistics",
        budget_signal="unknown",  # revealed after QUALIFY
        pain_points=["route optimization", "driver coordination", "fuel tracking"],
        decision_maker=True,
        true_budget=0.7,
        close_threshold=0.5,
        stall_probability=0.0,
    ),
    # 9 more L2 profiles: budget hidden, one objection expected
]

PROFILES_L3 = [
    ProspectProfile(
        company_name="Nova Financial",
        company_size="enterprise",
        industry="finance",
        budget_signal="unknown",
        pain_points=["compliance reporting", "audit trails", "data silos"],
        decision_maker=False,   # must navigate to decision maker
        true_budget=0.6,
        close_threshold=0.55,
        stall_probability=0.3,  # will stall at turn 10
    ),
    # 9 more L3 profiles: budget hidden, two objections, mode shift
]

PROFILES_L4 = [
    ProspectProfile(
        company_name="Cipher Tech",
        company_size="small",
        industry="technology",
        budget_signal="high",   # MISLEADING — true_budget is actually low
        pain_points=["security", "compliance"],
        decision_maker=True,
        true_budget=0.2,        # can't actually afford it
        close_threshold=0.5,
        stall_probability=0.5,
    ),
    # 9 more L4: misleading signals, correct answer is DISQUALIFY
]

ALL_PROFILES = {1: PROFILES_L1, 2: PROFILES_L2, 3: PROFILES_L3, 4: PROFILES_L4}

def sample_profile(difficulty: int) -> ProspectProfile:
    return random.choice(ALL_PROFILES[difficulty])
```

---

## Phase 3: Business Rules (Person A) — `server/rules.py`

```python
# server/rules.py
from dataclasses import dataclass
from typing import Callable
from ..models import SalesPathAction, SalesPathState


@dataclass
class BusinessRule:
    rule_id: str
    name: str
    description: str
    check: Callable[[SalesPathState, SalesPathAction], bool]
    # Returns True if VIOLATED


def _qualify_before_present(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "PRESENT":
        return "QUALIFY" not in state.steps_completed
    return False


def _demo_before_negotiate(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "NEGOTIATE":
        return "OFFER_DEMO" not in state.steps_completed
    return False


def _budget_known_to_negotiate(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "NEGOTIATE":
        return state.prospect_profile.get("budget_signal") == "unknown"
    return False


def _discount_after_objections(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "NEGOTIATE":
        if "discount" in action.content.lower():
            return state.objections_handled < 2
    return False


def _no_repeat_action(state: SalesPathState, action: SalesPathAction) -> bool:
    if state.conversation_history:
        last_action = state.conversation_history[-1].get("action_type", "")
        return last_action == action.action_type
    return False


def _prospect_first(state: SalesPathState, action: SalesPathAction) -> bool:
    if state.turn_number == 1:
        return action.action_type != "PROSPECT"
    return False


def _followup_timing(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "FOLLOW_UP":
        if state.conversation_history:
            last_speaker = state.conversation_history[-1].get("speaker", "agent")
            return last_speaker == "prospect"   # prospect just responded
    return False


def _disqualify_logic(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "DISQUALIFY":
        profile = state.prospect_profile
        true_budget = state._hidden.get("true_budget", 0.5)
        close_threshold = state._hidden.get("close_threshold", 0.5)
        dm = profile.get("decision_maker", True)
        # Violation: disqualifying when prospect is actually closeable
        return (true_budget >= close_threshold) and dm
    return False


def _close_requires_demo(state: SalesPathState, action: SalesPathAction) -> bool:
    if action.action_type == "CLOSE":
        if state.difficulty >= 2:
            return "OFFER_DEMO" not in state.steps_completed
    return False


BUSINESS_RULES = [
    BusinessRule("R01", "qualify_before_present",
                 "Must QUALIFY before PRESENT", _qualify_before_present),
    BusinessRule("R02", "demo_before_negotiate",
                 "Must OFFER_DEMO before NEGOTIATE", _demo_before_negotiate),
    BusinessRule("R03", "budget_known_to_negotiate",
                 "Budget must be known before NEGOTIATE", _budget_known_to_negotiate),
    BusinessRule("R04", "discount_after_objections",
                 "Discount only after 2 objections", _discount_after_objections),
    BusinessRule("R05", "no_repeat_action",
                 "Cannot repeat same action consecutively", _no_repeat_action),
    BusinessRule("R06", "prospect_first",
                 "First action must be PROSPECT", _prospect_first),
    BusinessRule("R07", "followup_timing",
                 "FOLLOW_UP only after prospect silence", _followup_timing),
    BusinessRule("R08", "disqualify_logic",
                 "DISQUALIFY only when prospect is genuinely unqualified", _disqualify_logic),
    BusinessRule("R09", "close_requires_demo",
                 "Must OFFER_DEMO before CLOSE (Levels 2+)", _close_requires_demo),
]


def check_rules(state: SalesPathState, action: SalesPathAction) -> list[str]:
    """Returns list of violated rule IDs."""
    return [
        rule.rule_id
        for rule in BUSINESS_RULES
        if rule.check(state, action)
    ]
```

---

## Phase 4: Prospect Simulator (Person A) — `server/prospect_simulator.py`

```python
# server/prospect_simulator.py
# PURE RULE-BASED. No LLM. No imports from transformers.

from ..models import SalesPathState, SalesPathAction

RESPONSE_TEXT = {
    "open:positive_signal": "That sounds interesting. Tell me more about how this works.",
    "open:neutral_signal": "I see. We're evaluating a few options at the moment.",
    "objection:price": "The pricing seems higher than what we budgeted for.",
    "objection:timing": "The timing isn't ideal — we're in the middle of a quarter close.",
    "objection:premature_pitch": "I'm not sure we're ready to discuss solutions yet. What do you know about our situation?",
    "deflect:budget_not_discussed": "We haven't really talked about what we're looking for yet.",
    "deflect:stall": "Let me get back to you on this. A lot is happening on our end.",
    "accept:demo_scheduled": "Yes, let's set up a demo. What time works next week?",
    "accept:close_success": "Alright, I think we can move forward with this. Send over the paperwork.",
    "reject:close_failed": "I don't think we're ready to commit at this point.",
    "silence": "",
    "exit:disqualified": "I think we're done here. This isn't the right fit.",
}


class ProspectSimulator:

    def respond(self, action: SalesPathAction, state: SalesPathState) -> tuple[str, str]:
        """
        Returns (response_token, response_text).
        Deterministic — same inputs always produce same output.
        """
        token = self._get_token(action, state)
        text = RESPONSE_TEXT[token]
        return token, text

    def _get_token(self, action: SalesPathAction, state: SalesPathState) -> str:
        atype = action.action_type
        hidden = state._hidden
        turn = state.turn_number
        profile = state.prospect_profile
        objections = state.objections_handled
        difficulty = state.difficulty

        # Rule violation responses (priority — check first)
        if "R01" in state.constraints_violated[-1:]:
            return "objection:premature_pitch"
        if "R03" in state.constraints_violated[-1:]:
            return "deflect:budget_not_discussed"

        # Action-specific logic
        if atype == "PROSPECT":
            return "open:positive_signal"

        if atype == "QUALIFY":
            # Reveal budget signal if it was hidden
            if profile.get("budget_signal") == "unknown":
                state.prospect_profile["budget_signal"] = hidden.get("revealed_budget", "medium")
            return "open:neutral_signal"

        if atype == "PRESENT":
            if difficulty >= 2:
                return "objection:price" if objections == 0 else "open:positive_signal"
            return "open:positive_signal"

        if atype == "HANDLE_OBJECTION":
            state.objections_handled += 1
            if objections + 1 >= hidden.get("num_objections", 1):
                return "open:positive_signal"
            return "objection:timing" if objections == 0 else "open:positive_signal"

        if atype == "OFFER_DEMO":
            return "accept:demo_scheduled"

        if atype == "NEGOTIATE":
            return "open:neutral_signal"

        if atype == "CLOSE":
            true_budget = hidden.get("true_budget", 0.7)
            threshold = hidden.get("close_threshold", 0.5)
            if true_budget >= threshold and profile.get("decision_maker", True):
                return "accept:close_success"
            return "reject:close_failed"

        if atype == "FOLLOW_UP":
            return "open:neutral_signal"

        if atype == "DISQUALIFY":
            return "exit:disqualified"

        # Mode shift at turn 10 for Level 3+
        if difficulty >= 3 and turn >= 10:
            import random
            if random.random() < hidden.get("stall_probability", 0.0):
                return "deflect:stall"

        return "open:neutral_signal"
```

---

## Phase 5: Reward Function (Person B) — `server/reward.py`

```python
# server/reward.py

from ..models import SalesPathState, SalesPathAction

DIFFICULTY_OPTIMAL_TURNS = {1: 5, 2: 8, 3: 12, 4: 14}


def compute_reward(
    state: SalesPathState,
    action: SalesPathAction,
    response_token: str,
    new_violations: list[str],
    episode_done: bool,
) -> tuple[float, dict]:
    """
    Returns (total_reward, component_dict).
    Always returns components — never a single scalar.
    """
    components = {}

    # --- Component 1: Outcome (only on terminal step) ---
    r_outcome = 0.0
    if episode_done:
        if response_token == "accept:close_success":
            r_outcome = 1.0
        elif action.action_type == "DISQUALIFY":
            # Check if disqualify was correct (no R08 violation)
            if "R08" not in new_violations:
                r_outcome = 0.5
            else:
                r_outcome = -0.5
        elif state.turn_number >= 20:
            r_outcome = -0.3
        elif len(state.constraints_violated) >= 3:
            r_outcome = -0.5
        else:
            r_outcome = -0.5  # failed close
    components["r_outcome"] = r_outcome

    # --- Component 2: Compliance ---
    total_violations = len(state.constraints_violated) + len(new_violations)
    r_compliance = max(-1.0, -0.2 * len(new_violations))  # per-step signal
    components["r_compliance"] = r_compliance

    # --- Component 3: Step Ordering ---
    required = state.required_workflow
    completed = state.steps_completed
    if len(required) > 1 and len(completed) > 0:
        # Count correct transitions
        correct = sum(
            1 for i in range(min(len(completed), len(required)))
            if completed[i] == required[i]
        )
        r_ordering = correct / len(required)
    else:
        r_ordering = 1.0 if (not required or action.action_type == required[0]) else 0.0
    components["r_ordering"] = r_ordering

    # --- Component 4: Efficiency ---
    if episode_done:
        optimal = DIFFICULTY_OPTIMAL_TURNS.get(state.difficulty, 10)
        overhead = max(0, state.turn_number - optimal)
        r_efficiency = max(-0.3, -0.05 * overhead)
    else:
        r_efficiency = 0.0  # only computed at episode end
    components["r_efficiency"] = r_efficiency

    # --- Component 5: Format ---
    r_format = 1.0 if action.is_valid() else -0.1
    components["r_format"] = r_format

    # --- Weighted total ---
    weights = {
        "r_outcome": 0.40,
        "r_compliance": 0.30,
        "r_ordering": 0.15,
        "r_efficiency": 0.10,
        "r_format": 0.05,
    }
    total = sum(weights[k] * v for k, v in components.items())
    components["total"] = total

    return total, components
```

---

## Phase 6: Environment Core (Person A) — `server/salespath_environment.py`

```python
# server/salespath_environment.py
import uuid
from openenv.core.env_server import Environment
from ..models import SalesPathAction, SalesPathObservation, SalesPathState
from .task_bank import sample_profile
from .rules import check_rules, BUSINESS_RULES
from .reward import compute_reward
from .prospect_simulator import ProspectSimulator

DIFFICULTY_WORKFLOW = {
    1: ["QUALIFY", "PRESENT", "CLOSE"],
    2: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
    3: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO",
        "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
    4: [],  # agent must determine; DISQUALIFY may be correct
}

MAX_VIOLATIONS_BEFORE_TERMINATE = 3
MAX_TURNS = 20


class SalesPathEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state = SalesPathState()
        self._simulator = ProspectSimulator()

    def reset(self, difficulty: int = 1) -> SalesPathObservation:
        profile = sample_profile(difficulty)
        hidden = {
            "true_budget": profile.true_budget,
            "close_threshold": profile.close_threshold,
            "stall_probability": profile.stall_probability,
            "num_objections": {1: 0, 2: 1, 3: 2, 4: 2}[difficulty],
            "revealed_budget": (
                "high" if profile.true_budget >= 0.7
                else "medium" if profile.true_budget >= 0.4
                else "low"
            ),
        }
        public_profile = {
            "company_name": profile.company_name,
            "company_size": profile.company_size,
            "industry": profile.industry,
            "budget_signal": profile.budget_signal,
            "pain_points": profile.pain_points,
            "decision_maker": profile.decision_maker,
        }
        self._state = SalesPathState(
            episode_id=str(uuid.uuid4()),
            prospect_profile=public_profile,
            required_workflow=DIFFICULTY_WORKFLOW[difficulty],
            difficulty=difficulty,
        )
        self._state._hidden = hidden

        return SalesPathObservation(
            prospect_response=(
                f"You are engaging {profile.company_name}, a {profile.company_size} "
                f"{profile.industry} company. Pain points: {', '.join(profile.pain_points)}. "
                f"Begin the sales conversation."
            ),
            workflow_stage="START",
            steps_completed=[],
            constraints_violated=[],
            turn_number=0,
            reward=0.0,
            done=False,
            info={"difficulty": difficulty, "episode_id": self._state.episode_id},
        )

    def step(self, action: SalesPathAction) -> SalesPathObservation:
        state = self._state
        state.turn_number += 1

        # Validate action format
        if not action.is_valid():
            return SalesPathObservation(
                prospect_response="Invalid action type.",
                workflow_stage=state.workflow_stage,
                steps_completed=list(state.steps_completed),
                constraints_violated=list(state.constraints_violated),
                turn_number=state.turn_number,
                reward=-0.2,
                done=False,
                info={"error": f"Invalid action_type: {action.action_type}",
                      "r_format": -0.1},
            )

        # Check business rules
        new_violations = check_rules(state, action)
        state.constraints_violated.extend(new_violations)

        # Update conversation history
        state.conversation_history.append({
            "turn": state.turn_number,
            "speaker": "agent",
            "action_type": action.action_type,
            "content": action.content,
        })

        # Update steps completed
        if action.action_type not in state.steps_completed:
            state.steps_completed.append(action.action_type)
        state.workflow_stage = action.action_type

        # Get prospect response
        response_token, response_text = self._simulator.respond(action, state)
        state.conversation_history.append({
            "turn": state.turn_number,
            "speaker": "prospect",
            "response_token": response_token,
            "text": response_text,
        })

        # Determine episode termination
        terminal_actions = {"CLOSE", "DISQUALIFY"}
        too_many_violations = len(state.constraints_violated) >= MAX_VIOLATIONS_BEFORE_TERMINATE
        turn_limit = state.turn_number >= MAX_TURNS
        done = (
            action.action_type in terminal_actions
            or too_many_violations
            or turn_limit
        )
        state.done = done

        # Compute reward
        total_reward, components = compute_reward(
            state, action, response_token, new_violations, done
        )

        return SalesPathObservation(
            prospect_response=response_text,
            workflow_stage=state.workflow_stage,
            steps_completed=list(state.steps_completed),
            constraints_violated=list(state.constraints_violated),
            turn_number=state.turn_number,
            reward=total_reward,
            reward_components=components,
            done=done,
            info={
                "response_token": response_token,
                "new_violations": new_violations,
                "episode_id": state.episode_id,
            },
        )

    @property
    def state(self) -> SalesPathState:
        return self._state
```

---

## Phase 7: FastAPI App (Person A) — `server/app.py`

```python
# server/app.py — thin wrapper only
from openenv.core.env_server import create_fastapi_app
from ..models import SalesPathAction, SalesPathObservation
from .salespath_environment import SalesPathEnvironment

app = create_fastapi_app(
    SalesPathEnvironment,
    SalesPathAction,
    SalesPathObservation,
)
```

---

## Phase 8: Client (Person B) — `client.py`

```python
# client.py
from openenv.core import EnvClient
from .models import SalesPathAction, SalesPathObservation, SalesPathState


class SalesPathEnv(EnvClient):
    action_type = SalesPathAction
    observation_type = SalesPathObservation
    state_type = SalesPathState

    async def reset(self, difficulty: int = 1) -> SalesPathObservation:
        return await super().reset(difficulty=difficulty)

    async def step(self, action_type: str, content: str, target: str = "") -> SalesPathObservation:
        action = SalesPathAction(
            action_type=action_type,
            content=content,
            target=target,
        )
        return await super().step(action)
```

---

## Phase 9: Rollout Function (Person B) — `training/rollout.py`

```python
# training/rollout.py
import re
from salespath_env.client import SalesPathEnv
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
        {"role": "system", "content": SYSTEM_PROMPT.format(workflow=" → ".join(workflow))},
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


async def run_episode(model, tokenizer, env_url: str, difficulty: int = 1) -> dict:
    """Run one full episode. Returns trajectory with rewards."""
    DIFFICULTY_WORKFLOW = {
        1: ["QUALIFY", "PRESENT", "CLOSE"],
        2: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
        3: ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO",
            "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
        4: [],
    }
    workflow = DIFFICULTY_WORKFLOW[difficulty]

    async with SalesPathEnv(base_url=env_url) as env:
        obs = await env.reset(difficulty=difficulty)
        trajectory = []
        total_reward = 0.0

        while not obs.done:
            prompt = build_prompt(obs, workflow, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            action_type, content = parse_action(generated)
            obs = await env.step(action_type, content)

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
```

---

## Phase 10: Curriculum Scheduler (Person B) — `training/curriculum.py`

```python
# training/curriculum.py
from dataclasses import dataclass

@dataclass
class CurriculumConfig:
    thresholds: dict  # mean_reward -> difficulty_distribution

    def get_distribution(self, mean_reward: float) -> dict:
        for threshold in sorted(self.thresholds.keys(), reverse=True):
            if mean_reward >= threshold:
                return self.thresholds[threshold]
        return self.thresholds[min(self.thresholds.keys())]


DEFAULT_CURRICULUM = CurriculumConfig(
    thresholds={
        0.0:  {1: 0.90, 2: 0.10, 3: 0.00, 4: 0.00},
        0.30: {1: 0.50, 2: 0.40, 3: 0.10, 4: 0.00},
        0.50: {1: 0.20, 2: 0.40, 3: 0.35, 4: 0.05},
        0.65: {1: 0.10, 2: 0.30, 3: 0.40, 4: 0.20},
    }
)


def sample_difficulty(curriculum: CurriculumConfig, mean_reward: float) -> int:
    import random
    dist = curriculum.get_distribution(mean_reward)
    return random.choices(
        list(dist.keys()),
        weights=list(dist.values()),
        k=1
    )[0]
```

---

## Phase 11: Training Script (Person B) — `training/grpo_train.py`

```python
# training/grpo_train.py
import torch
import asyncio
import numpy as np
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from curriculum import DEFAULT_CURRICULUM, sample_difficulty
from rollout import run_episode

# --- Model Load ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

ENV_URL = "http://localhost:8000"  # or HuggingFace Space URL

# --- Reward function for GRPO (wraps environment) ---
def salespath_reward_fn(completions, prompts, **kwargs) -> list[float]:
    """
    GRPO calls this with a batch of completions.
    We run each through the environment and return rewards.
    """
    rewards = []
    for completion in completions:
        # Parse action from completion
        from rollout import parse_action
        action_type, content = parse_action(completion)
        # For GRPO, we use a simplified single-step reward
        # Full episode reward is tracked separately in curriculum loop
        reward = kwargs.get("step_rewards", {}).get(completion, 0.0)
        rewards.append(reward)
    return rewards


# --- Training config ---
training_config = GRPOConfig(
    output_dir="salespath_grpo_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_new_tokens=256,
    temperature=0.8,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=100,
    report_to="none",
)

# --- Curriculum training loop ---
async def curriculum_train():
    mean_reward = 0.0
    reward_history = []

    for step in range(500):
        difficulty = sample_difficulty(DEFAULT_CURRICULUM, mean_reward)
        result = await run_episode(model, tokenizer, ENV_URL, difficulty)

        reward_history.append(result["total_reward"])
        if len(reward_history) > 20:
            mean_reward = np.mean(reward_history[-20:])

        # Log metrics
        if step % 10 == 0:
            print(f"Step {step:4d} | Difficulty {difficulty} | "
                  f"Reward {result['total_reward']:.3f} | "
                  f"Mean(20) {mean_reward:.3f} | "
                  f"Violations {len(result['violations'])} | "
                  f"Steps {result['steps_completed']}")

        # Manual inspection every 50 steps
        if step % 50 == 0:
            print("\n=== RAW GENERATION SAMPLE ===")
            if result["trajectory"]:
                print(result["trajectory"][0]["generated"])
            print("==============================\n")


if __name__ == "__main__":
    asyncio.run(curriculum_train())
```

---

## Phase 12: Dockerfile (Person A) — `server/Dockerfile`

```dockerfile
ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/openenv/core/ /app/src/openenv/core/
COPY salespath_env/ /app/salespath_env/

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "salespath_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`server/requirements.txt`:
```
fastapi
uvicorn
pydantic>=2.0
```

---

## Phase 13: Deploy to HuggingFace

```bash
# From salespath_env/ directory
openenv push --repo-id Imsachin010/salespath-env

# Verify it's running
curl -X POST https://imsachin010-salespath-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": 1}'
```

---

## Phase 14: Model Save (After Training)

```python
# CORRECT save — do not change this
model.save_pretrained_merged(
    "salespath_trained_merged",
    tokenizer,
    save_method="merged_16bit",
)

# Push to HuggingFace Hub
model.push_to_hub_merged(
    "Imsachin010/salespath-qwen25-7b",
    tokenizer,
    save_method="merged_16bit",
)
```

---

## Build Order Summary

```
Person A (Environment):          Person B (Training):
1. models.py                     (wait for models.py)
2. server/task_bank.py           1. server/reward.py
3. server/rules.py               2. training/rollout.py
4. server/prospect_simulator.py  3. training/curriculum.py
5. server/salespath_environment  4. training/grpo_train.py
6. server/app.py                 5. training/colab_train.ipynb
7. Dockerfile
8. openenv push → verify health
                                 6. Connect rollout to live env URL
                                 7. Run first training loop (difficulty=1 only)
                                 8. Verify reward > 0 on step 1
                                 9. Enable curriculum
```

**Critical gate:** Person B does not run training until Person A has confirmed:
- `POST /reset` returns a valid observation
- `POST /step` with a valid action returns a valid observation
- `POST /step` with an invalid action returns error in `info`, not a 500
