# SalesPath — Agent Rules & Constraints
### Read this before touching any file. These are non-negotiable.

---

## 0. Project Identity

- **Project name:** `salespath_env`
- **HuggingFace repo:** `Imsachin010/salespath-env`
- **Theme:** Theme #2 — Long-Horizon Planning (Scale AI bonus prize)
- **Stack:** OpenEnv + GRPO (HF TRL) + Unsloth + Qwen 2.5 7B Instruct

---

## 1. Directory Structure — Do Not Deviate

```
salespath_env/
├── __init__.py
├── models.py              ← ALL Pydantic dataclasses live here only
├── client.py              ← SalesPathEnv(EnvClient) lives here only
├── README.md
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── __init__.py
│   ├── salespath_environment.py   ← SalesPathEnvironment(Environment)
│   ├── prospect_simulator.py      ← ProspectSimulator (rule-based only)
│   ├── reward.py                  ← ALL reward logic lives here only
│   ├── task_bank.py               ← ALL prospect profiles and tasks
│   ├── rules.py                   ← ALL business rule definitions
│   ├── app.py                     ← FastAPI app only, no logic
│   ├── requirements.txt
│   └── Dockerfile
training/
├── grpo_train.py          ← training script
├── rollout.py             ← rollout function
├── curriculum.py          ← difficulty scheduler
└── colab_train.ipynb      ← Colab notebook for judges
```

---

## 2. OpenEnv API — Exact Signatures to Follow

```python
# models.py — extend these base classes
from openenv.core import Action, Observation, State  # actual imports

class SalesPathAction(Action):
    action_type: str       # one of the 9 valid action types
    content: str           # natural language content of the action
    target: str = ""       # optional target (e.g., which objection)

class SalesPathObservation(Observation):
    prospect_response: str
    workflow_stage: str
    constraints_violated: list[str]
    steps_completed: list[str]
    turn_number: int
    reward: float
    done: bool
    info: dict

class SalesPathState(State):
    episode_id: str
    prospect_profile: dict
    conversation_history: list[dict]
    workflow_stage: str
    steps_completed: list[str]
    constraints_violated: list[str]
    turn_number: int
    difficulty: int        # 1, 2, 3, or 4
    hidden_state: dict     # NOT exposed to agent
```

```python
# server/salespath_environment.py
from openenv.core.env_server import Environment

class SalesPathEnvironment(Environment):
    def reset(self, difficulty: int = 1) -> SalesPathObservation: ...
    def step(self, action: SalesPathAction) -> SalesPathObservation: ...
    @property
    def state(self) -> SalesPathState: ...
```

```python
# server/app.py — nothing else in this file
from openenv.core.env_server import create_fastapi_app
from ..models import SalesPathAction, SalesPathObservation
from .salespath_environment import SalesPathEnvironment

app = create_fastapi_app(SalesPathEnvironment, SalesPathAction, SalesPathObservation)
```

---

## 3. Hard Rules — Code Will Be Rejected If Violated

### 3.1 No LLM in the Environment
- `ProspectSimulator` is a **pure rule-based state machine**
- No API calls, no model inference, no `transformers` imports inside `server/`
- If you find yourself writing `model.generate()` inside `server/`, stop. Wrong file.

### 3.2 Immutable Prospect State
- Once `reset()` sets the prospect profile, agent actions **cannot modify `hidden_state`**
- `hidden_state` is read-only after `reset()`
- Never expose `hidden_state` fields in `SalesPathObservation`

### 3.3 Reward Lives in One Place
- All reward computation goes in `server/reward.py`
- `salespath_environment.py` calls `compute_reward()` — it does not compute reward itself
- Never compute reward inside `step()` directly

### 3.4 Business Rules Live in One Place
- All rule definitions go in `server/rules.py` as a list of `BusinessRule` dataclasses
- `step()` calls `check_rules(state, action)` from `rules.py` — it does not check rules inline

### 3.5 Turn Limit is Absolute
- Max turns = 20. Hard terminate. No exceptions.
- Episode must set `done=True` and assign `r_outcome = -0.3` at turn 20 regardless of state

### 3.6 Action Validation is Strict
- If `action_type` is not one of the 9 valid types, return `done=False`, `reward=-0.2`, observation with error message
- Do not raise exceptions to the agent — return a valid `SalesPathObservation` with error in `info`

### 3.7 Reward Must Be Multi-Component
- Reward function must log all 5 components separately in `info` dict
- Never return a single scalar reward without component breakdown
- Component keys: `r_outcome`, `r_compliance`, `r_ordering`, `r_efficiency`, `r_format`

### 3.8 No Global Mutable State in Environment
- Each WebSocket session gets its own `SalesPathEnvironment` instance
- No class-level variables that change during episodes
- No module-level state

---

## 4. Valid Action Types — Exact Strings

```python
VALID_ACTIONS = {
    "PROSPECT",        # initial outreach — only valid on turn 1
    "QUALIFY",         # ask qualification questions
    "PRESENT",         # deliver pitch
    "HANDLE_OBJECTION", # respond to raised objection
    "OFFER_DEMO",      # propose product demonstration
    "NEGOTIATE",       # discuss pricing/terms
    "CLOSE",           # submit closing offer → terminates episode
    "FOLLOW_UP",       # follow up after no response
    "DISQUALIFY",      # exit if prospect is not a fit → terminates episode
}
```

---

## 5. Business Rules — Exact Definitions

These are checked after every `step()`. Each violation increments `constraints_violated`.

```python
RULES = [
    # ID   Name                      Condition for VIOLATION
    R01  "qualify_before_present"    PRESENT called before any QUALIFY
    R02  "demo_before_negotiate"     NEGOTIATE called before OFFER_DEMO
    R03  "budget_known_to_negotiate" NEGOTIATE called while budget_signal == "unknown"
    R04  "discount_after_objections" Discount mentioned in NEGOTIATE before 2 objections handled
    R05  "no_repeat_action"          Same action_type on consecutive turns
    R06  "prospect_first"            Any action other than PROSPECT on turn 1
    R07  "followup_timing"           FOLLOW_UP called when prospect responded last turn
    R08  "disqualify_logic"          DISQUALIFY called when budget >= threshold AND decision_maker==True
    R09  "close_requires_demo"       CLOSE called before OFFER_DEMO
]
```

Three violations → `done=True`, `r_outcome = -0.5`

---

## 6. Prospect Simulator — Exact Response Rules

`ProspectSimulator.respond(action, state)` returns one of these string tokens. The environment converts tokens to natural language text for the observation.

```python
RESPONSE_TOKENS = {
    "open:positive_signal",      # prospect is engaged and open
    "open:neutral_signal",       # prospect acknowledges but non-committal
    "objection:price",           # raises price objection
    "objection:timing",          # raises timing objection
    "objection:premature_pitch", # triggered by R01 violation
    "deflect:budget_not_discussed", # triggered by R03 violation
    "deflect:stall",             # prospect stalls (Level 3+)
    "accept:demo_scheduled",     # agrees to demo
    "accept:close_success",      # agrees to close → episode success
    "reject:close_failed",       # rejects close
    "silence",                   # no response (enables FOLLOW_UP)
    "exit:disqualified",         # prospect exits conversation
}
```

---

## 7. Difficulty Configuration

```python
DIFFICULTY_CONFIG = {
    1: {
        "max_turns": 20,
        "workflow_steps": ["QUALIFY", "PRESENT", "CLOSE"],
        "num_objections": 0,
        "budget_hidden": False,
        "mode_shift": False,
        "optimal_turns": 5,
    },
    2: {
        "max_turns": 20,
        "workflow_steps": ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO", "CLOSE"],
        "num_objections": 1,
        "budget_hidden": True,  # revealed after QUALIFY
        "mode_shift": False,
        "optimal_turns": 8,
    },
    3: {
        "max_turns": 20,
        "workflow_steps": ["QUALIFY", "PRESENT", "HANDLE_OBJECTION", "OFFER_DEMO",
                           "HANDLE_OBJECTION", "NEGOTIATE", "CLOSE"],
        "num_objections": 2,
        "budget_hidden": True,
        "mode_shift": True,    # prospect signals shift at turn 10
        "optimal_turns": 12,
    },
    4: {
        "max_turns": 20,
        "workflow_steps": "full",  # agent must determine correct path
        "num_objections": 2,
        "budget_hidden": True,
        "mode_shift": True,
        "misleading_signals": True,  # budget signals are deceptive
        "optimal_turns": 14,
    },
}
```

---

## 8. Reward — Exact Weights

```python
REWARD_WEIGHTS = {
    "r_outcome":    0.40,
    "r_compliance": 0.30,
    "r_ordering":   0.15,
    "r_efficiency": 0.10,
    "r_format":     0.05,
}

OUTCOME_VALUES = {
    "close_success":         1.0,
    "disqualify_correct":    0.5,
    "turn_limit_reached":   -0.3,
    "close_failed":         -0.5,
    "three_violations":     -0.5,
}

COMPLIANCE_PER_VIOLATION = -0.2   # capped at -1.0
EFFICIENCY_PER_EXTRA_TURN = -0.05 # capped at -0.3
FORMAT_PASS = 1.0
FORMAT_FAIL = -0.1
```

---

## 9. Training Rules

### Prompt Format (what gets sent to the LLM)
```
System: You are a B2B sales agent. Follow this workflow strictly:
{workflow_steps_for_difficulty}

Business rules you must never violate:
{rules_list}

Current state:
- Prospect: {prospect_summary}
- Stage: {workflow_stage}
- Steps done: {steps_completed}
- Turn: {turn_number}/20

Prospect said: {prospect_response}

Respond with:
ACTION: <action_type>
CONTENT: <your message>
```

### Response parsing
- Extract `ACTION:` line → `action_type`
- Extract `CONTENT:` line → `content`
- If parsing fails → `r_format = -0.1`, use fallback QUALIFY

### GRPO config
```python
GRPOConfig(
    num_generations=8,        # rollouts per prompt
    max_new_tokens=256,
    temperature=0.8,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)
```

---

## 10. What to Monitor During Training

Log these every 10 steps. If any of these goes wrong, stop and inspect raw generations:

| Metric | Healthy Range | Alarm |
|--------|--------------|-------|
| `mean_reward` | Rising | Flat for >50 steps |
| `mean_r_compliance` | Rising | < -0.5 after step 100 |
| `violations_per_episode` | Falling | > 3.0 after step 100 |
| `ordering_rate` | Rising toward 0.85 | < 0.3 after step 150 |
| `close_success_rate` | Rising | 0 after step 200 |

Inspect raw generations every 50 steps. Look for: repeated actions, empty CONTENT, invalid ACTION types, CLOSE before QUALIFY.

---

## 11. Save Model Correctly

```python
# CORRECT — do not deviate
model.save_pretrained_merged(
    "salespath_trained",
    tokenizer,
    save_method="merged_16bit",  # NOT naive upcast of 4bit
)
```

Never do: `model.save_pretrained()` on a 4-bit model without merging first.

---

## 12. File Ownership (2-Person Team)

| Person | Files |
|--------|-------|
| **A** | `models.py`, `server/salespath_environment.py`, `server/prospect_simulator.py`, `server/rules.py`, `server/task_bank.py`, `server/app.py`, `Dockerfile` |
| **B** | `server/reward.py`, `training/grpo_train.py`, `training/rollout.py`, `training/curriculum.py`, `training/colab_train.ipynb`, `client.py` |

Both: `README.md`, `openenv.yaml`, `pyproject.toml`
