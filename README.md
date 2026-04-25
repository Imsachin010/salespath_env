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

# SalesPath Environment

A [OpenEnv](https://github.com/openenv)-compatible Reinforcement Learning gym environment for training sales agents via LLM fine-tuning.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset the environment, returns initial observation |
| `POST` | `/step` | Take an action, returns next observation + reward |
| `GET`  | `/health` | Health check |

## Quick Start

### Reset
```bash
curl -X POST https://imsachin010-salespath-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": 1}'
```

### Step
```bash
curl -X POST https://imsachin010-salespath-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "PROSPECT", "content": "Hello, tell me about your workflow challenges."}}'
```

## Action Types

- `PROSPECT` — Initial outreach and discovery
- `QUALIFY` — Qualify the lead
- `PRESENT` — Deliver the sales pitch
- `HANDLE_OBJECTION` — Handle prospect objections
- `OFFER_DEMO` — Offer product demonstration
- `NEGOTIATE` — Discuss pricing and terms
- `FOLLOW_UP` — Follow-up message
- `DISQUALIFY` — Exit if prospect is not a fit
- `CLOSE` — Attempt to close the deal
