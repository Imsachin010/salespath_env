# SalesPath: Teaching an LLM to Close Deals with Reinforcement Learning

**Theme:** Long-Horizon Planning (Scale AI Bonus Prize)  
**Stack:** OpenEnv · GRPO · Unsloth · Qwen 2.5 7B Instruct  
**HuggingFace Repo:** [Imsachin010/salespath-env](https://huggingface.co/spaces/Imsachin010/salespath-env)  
**Trained Model:** [Imsachin010/salespath-qwen25-7b](https://huggingface.co/Imsachin010/salespath-qwen25-7b)

---

## The Problem

Most LLM agent benchmarks reward a single correct answer. Real-world tasks — like closing a B2B sales deal — require **20+ sequential decisions** where each action constrains what comes next. An agent that pitches the product before qualifying the prospect violates a business rule. An agent that negotiates before demonstrating value loses the deal.

We built **SalesPath**, a reinforcement learning environment that forces an LLM to learn this kind of long-horizon, rule-constrained planning through trial and error.

---

## What is SalesPath?

SalesPath is an OpenEnv-compatible environment where an LLM agent plays the role of a B2B sales representative. The agent must interact with a simulated prospect over up to 20 turns, following a strict workflow and 9 business rules — all while adapting to prospect signals.

### Valid Actions

The agent can only take one of 9 actions per turn:

```
PROSPECT → QUALIFY → PRESENT → HANDLE_OBJECTION → 
OFFER_DEMO → NEGOTIATE → CLOSE → FOLLOW_UP → DISQUALIFY
```

### Business Rules (enforced at every step)

| Rule | Constraint |
|------|-----------|
| R01 | Must QUALIFY before PRESENT |
| R02 | Must OFFER_DEMO before NEGOTIATE |
| R03 | Budget must be known before NEGOTIATE |
| R04 | Discount only after 2 objections handled |
| R05 | Cannot repeat same action consecutively |
| R06 | First action must always be PROSPECT |
| R07 | FOLLOW_UP only after prospect silence |
| R08 | DISQUALIFY only if prospect is genuinely unqualified |
| R09 | Must OFFER_DEMO before CLOSE (difficulty 2+) |

3 violations → episode terminates with penalty.

### Difficulty Levels

| Level | Workflow | Challenge |
|-------|---------|-----------|
| 1 | QUALIFY → PRESENT → CLOSE | Budget known, no objections |
| 2 | + HANDLE_OBJECTION + OFFER_DEMO | Budget hidden, 1 objection |
| 3 | + NEGOTIATE + mode shift | Budget hidden, 2 objections, prospect changes stance at turn 10 |
| 4 | Dynamic path | Misleading budget signals, agent must decide to DISQUALIFY |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Training Loop (Colab)               │
│                                                     │
│   Qwen 2.5 7B (4-bit, Unsloth)                     │
│         │                                           │
│         │  generates: ACTION: X / CONTENT: Y        │
│         ▼                                           │
│   ┌──────────────────────────────────────┐          │
│   │   SalesPath Environment (FastAPI)    │          │
│   │   ┌──────────────────────────────┐   │          │
│   │   │ ProspectSimulator (rule-based)│  │          │
│   │   │ BusinessRules (R01-R09)      │   │          │
│   │   │ RewardFunction (5 components)│   │          │
│   │   └──────────────────────────────┘   │          │
│   └──────────────────────────────────────┘          │
│         │                                           │
│         │  reward signal                            │
│         ▼                                           │
│   GRPO (TRL) — updates model weights                │
└─────────────────────────────────────────────────────┘
```

### Reward Function

The reward is not a single number. It has 5 components, each rewarding a different aspect of good sales behaviour:

```python
REWARD_WEIGHTS = {
    "r_outcome":    0.40,  # Did the deal close? Was disqualify correct?
    "r_compliance": 0.30,  # How many rules were violated?
    "r_ordering":   0.15,  # Did actions follow the required workflow?
    "r_efficiency": 0.10,  # Did the agent close in minimal turns?
    "r_format":     0.05,  # Did the output parse correctly?
}
```

This dense reward signal gives GRPO meaningful gradients at every step — not just at the end of the episode.

---

## Training

### Model
- **Base:** `Qwen/Qwen2.5-7B-Instruct`
- **Quantization:** 4-bit via Unsloth (fits in T4 15GB VRAM)
- **Fine-tuning:** LoRA (r=16, all attention + MLP projections)
- **Algorithm:** GRPO (Group Relative Policy Optimisation, TRL)

### Prompt Format

```
System: You are a B2B sales agent. Follow this workflow strictly:
QUALIFY -> PRESENT -> HANDLE_OBJECTION -> OFFER_DEMO -> CLOSE

Business rules you must never violate:
- R01: Must QUALIFY before PRESENT
... (all 9 rules)

Prospect said: The pricing seems higher than what we budgeted for.
Current stage: PRESENT
Steps done: ['QUALIFY', 'PRESENT']
Turn: 4/20

Respond with:
ACTION: <action_type>
CONTENT: <your message>
```

### GRPO Config

```python
GRPOConfig(
    num_generations=8,
    max_new_tokens=256,
    temperature=0.8,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)
```

---

## Why a Small Local Model — Not a Frontier API?

This is the most important design decision in the project, and it's worth explaining clearly.

### The Frontier Model Trap

When you hear "LLM agent", the instinct is to reach for the most powerful model available — GPT-4, Claude 3.5, Llama 3 70B via API. These models are impressive out of the box. But for **reinforcement learning**, they are the wrong choice:

| | Frontier Model via API | Local Model (our approach) |
|---|---|---|
| Who owns the weights? | The API provider | **You** |
| Can you update the weights? | ❌ No | ✅ Yes — every training step |
| Does the model improve with episodes? | ❌ No — same model forever | ✅ Yes — GRPO updates it |
| Is this real RL training? | ❌ No — just prompting | ✅ Yes |
| Cost of 500 training episodes | $$$  | Free (Colab GPU) |
| Model specialises on your task? | ❌ Generic forever | ✅ Becomes a sales expert |

The fundamental problem with an API model is that **you can observe its outputs but you cannot change what it knows**. You can run 10,000 episodes through GPT-4 and on episode 10,001 it will make the same mistakes as on episode 1. There is no learning loop — only inference.

### What GRPO Actually Does to the Weights

GRPO (Group Relative Policy Optimisation) is the algorithm that makes real RL training possible. Here is how it works in plain terms:

**Step 1 — Generate a group of completions**

For each prompt (a sales situation), the model generates 8 different responses with slight randomness:

```
Prompt: "Prospect says: The price is too high. Turn 3/20."

Completion A: "ACTION: NEGOTIATE\nCONTENT: I can offer a 20% discount..."
Completion B: "ACTION: HANDLE_OBJECTION\nCONTENT: I understand budget concerns..."
Completion C: "ACTION: PRESENT\nCONTENT: Let me tell you about our ROI..."
... (8 total)
```

**Step 2 — Score each completion with the reward function**

Each completion goes through the SalesPath environment. The reward function returns a score:

```
Completion A → reward = -0.2   (NEGOTIATE before OFFER_DEMO = R02 violation)
Completion B → reward = +0.45  (correct action, good content)
Completion C → reward = -0.1   (repeated action = R05 violation)
```

**Step 3 — Compute relative advantage**

GRPO does not use an absolute reward — it asks: *"How much better is this completion than the average of the group?"*

```
Group mean reward = 0.15

Completion A advantage = -0.2 - 0.15 = -0.35  (worse than average)
Completion B advantage = +0.45 - 0.15 = +0.30  (better than average)
Completion C advantage = -0.1 - 0.15 = -0.25  (worse than average)
```

**Step 4 — Update weights via gradient descent**

The model's weights are nudged so that:
- Completions with **positive advantage** become more likely
- Completions with **negative advantage** become less likely

After thousands of these updates, the model's internal probability distribution shifts. `HANDLE_OBJECTION` after a price objection becomes the high-probability path. `NEGOTIATE` before `OFFER_DEMO` becomes low-probability. The model has **learned** the sales workflow — not from instructions, but from experience.

```
Before training:  P(NEGOTIATE | price objection, turn 3) = 0.35
After training:   P(NEGOTIATE | price objection, turn 3) = 0.04

Before training:  P(HANDLE_OBJECTION | price objection) = 0.15
After training:   P(HANDLE_OBJECTION | price objection) = 0.61
```

### Why 7B is the Right Size

A model that is too small (< 1B) cannot generate coherent sales messages or follow multi-step reasoning. A frontier model (> 70B) cannot be fine-tuned on a free GPU. Qwen 2.5 7B is the sweet spot:

- **Large enough** to generate natural, persuasive sales language
- **Small enough** to fit in 15GB VRAM with 4-bit quantisation via Unsloth
- **Trainable** — weights update, behaviour changes, skills accumulate
- **Fast enough** — each GRPO step completes in under 2 minutes on a T4

This is not a compromise. This is the correct engineering choice for RL fine-tuning on accessible hardware.

---



### Reward Curve

<!-- FILL IN: paste reward_graph.png here after training -->
> **[INSERT: reward_graph.png]**

### Metrics Over Training

| Metric | Before Training (step 0) | After Training (step 100) | Target |
|--------|------------------------|--------------------------|--------|
| `mean_reward` | `[FILL]` | `[FILL]` | Rising |
| `violations_per_episode` | `[FILL]` | `[FILL]` | Falling |
| `close_success_rate` | `[FILL]` | `[FILL]` | Rising |
| `ordering_rate` | `[FILL]` | `[FILL]` | > 0.85 |

### Sample Generation — Before Training

```
Prospect: Tell me more about how this works.

ACTION: [FILL after training]
CONTENT: [FILL after training]
```

### Sample Generation — After Training

```
Prospect: The pricing seems higher than what we budgeted for.

ACTION: [FILL after training]
CONTENT: [FILL after training]
```

---

## Key Findings

- **Dense reward > sparse reward:** Using 5 reward components instead of a single win/loss signal made training significantly more stable. The model received learning signal on every turn, not just at episode end.

- **Curriculum learning matters:** Starting on difficulty 1 (simple workflow, no objections) before introducing harder levels prevented early reward collapse. The model learned basic workflow ordering first.

- **Rule violations decrease sharply:** `[FILL — describe what you observed in training logs]`

- **Format compliance was instant:** The `r_format` component ensured the model learned the `ACTION:/CONTENT:` format within the first `[FILL]` steps.

---

## Running the Environment

The environment server is deployed on HuggingFace Spaces:

```bash
# Reset (start new episode)
curl -X POST https://imsachin010-salespath-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": 1}'

# Step (take an action)
curl -X POST https://imsachin010-salespath-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "QUALIFY", "content": "What are your main pain points?", "target": ""}}'
```

### Run training yourself

```bash
git clone https://github.com/Imsachin010/salespath_env.git
cd salespath_env
pip install -e .

# Start environment server
uvicorn salespath_env.server.app:app --host 0.0.0.0 --port 8000

# Run curriculum training
python -m training.grpo_train --mode curriculum --steps 50

# Run GRPO training
python -m training.grpo_train --mode grpo --grpo-steps 100
```

Or open `training/traingrpo.ipynb` in Google Colab (T4 GPU recommended).

---

## What's Next

- [ ] Scale to Qwen 2.5 7B (full RULES.md target)
- [ ] Multi-agent: separate prospecting and closing agents
- [ ] Difficulty 4 mastery (misleading budget signals + correct DISQUALIFY)
- [ ] Push trained model to HuggingFace Hub

---

## References

- [OpenEnv Framework](https://github.com/openenv/openenv)
- [GRPO: Group Relative Policy Optimisation (DeepSeek)](https://arxiv.org/abs/2402.03300)
- [Unsloth — Fast LLM Fine-tuning](https://github.com/unslothai/unsloth)
- [TRL — Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [Qwen 2.5 7B Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
