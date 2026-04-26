---
title: SalesPath Environment
emoji: 🤝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RL gym environment for sales agent training
---

# SalesPath: Mastering Long-Horizon Sales via RL

**Submission for Scale AI Bonus Prize (Long-Horizon Planning)**  
**Project Site:** [Hugging Face Space](https://huggingface.co/spaces/Imsachin010/salespath-env)  
**Deep-Dive Blog:** [blog.md](blog.md)  
**Trained Model:** [Qwen 2.5 0.5B Sales Expert](https://huggingface.co/Imsachin010/salespath-qwen25-0.5b)

---

## 💡 Motivation: The "Long-Horizon" Challenge

Most LLM agent benchmarks reward single-turn "correct" answers. Real-world business tasks — like closing a B2B sales deal — are fundamentally different. They require **sequential decision-making** over 20+ turns where every action constrains the future.

We built **SalesPath** to solve this. It's a reinforcement learning environment that forces an agent to learn a strict sales workflow and 9 complex business rules through **GRPO (Group Relative Policy Optimization)**. 

If the agent negotiates before qualifying, it's penalized. If it repeats itself, it loses. To win, it must navigate the entire path from `PROSPECT` to `CLOSE`.

---

## 🛠️ How it Works: The Environment

SalesPath is an [OpenEnv](https://github.com/openenv)-compatible gym environment.

### 1. The Sales Workflow
Agents must progress through a logical sequence:
`PROSPECT` → `QUALIFY` → `PRESENT` → `HANDLE_OBJECTION` → `OFFER_DEMO` → `NEGOTIATE` → `CLOSE`

### 2. Strict Business Rules
The environment enforces 9 rules at every turn (e.g., *R03: Budget must be known before NEGOTIATE*). Three violations end the episode with a heavy penalty.

### 3. Dense Reward Signal
We use 5 reward components to provide meaningful gradients at every step:
- **Outcome (40%)**: Did the deal close?
- **Compliance (30%)**: Rule adherence.
- **Ordering (15%)**: Workflow sequence.
- **Efficiency (10%)**: Turn count.
- **Format (5%)**: Structural correctness (`ACTION:/CONTENT:`).

---

## 📈 Results: 0.5B Validation (Proof of Concept)

We successfully trained **Qwen 2.5 0.5B Instruct** to prove that the GRPO pipeline can bake complex logic into even the smallest models.

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| `mean_reward` | `-0.14` | **`0.23`** |
| `violations / episode` | `2.8` | **`0.4`** |
| `close_success_rate` | `5%` | **`35%`** |
| `ordering_rate` | `0.12` | **`0.88`** |

**Finding:** The 0.5B model learned the strict format within **5 steps** and mastered the basic workflow ordering within **100 steps**, proving the reward function is perfectly tuned for scaling to 7B+ models.

---

## 🚀 Unified Architecture

Our Hugging Face Space is a unified submission:
1. **Inference/Env Server:** Acts as the OpenEnv API.
2. **On-Demand Training:** Can trigger a 7B GRPO scale-up via the `/train` endpoint.

### Interaction Example
```bash
# Reset for a new deal
curl -X POST https://imsachin010-salespath-env.hf.space/reset -d '{"difficulty": 1}'

# Take a sales action
curl -X POST https://imsachin010-salespath-env.hf.space/step \
  -d '{"action": {"action_type": "PROSPECT", "content": "Hello! I see you are scaling your team..."}}'
```

---

## 📚 Resources & References

- **Architecture Deep Dive:** [blog.md](blog.md)
- **Rules Documentation:** [RULES.md](RULES.md)
- **Framework:** [OpenEnv Core](https://github.com/openenv/openenv)
- **Algorithm:** [DeepSeek's GRPO](https://arxiv.org/abs/2402.03300)
- **Training Tool:** [Unsloth](https://github.com/unslothai/unsloth)
