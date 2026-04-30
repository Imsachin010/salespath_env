# SalesPath — Business Rules (R01–R09)

The environment enforces these 9 business rules at every step.  
Three violations → episode terminates with a heavy penalty.

---

## R01 — Qualify Before Present

> **Must QUALIFY before PRESENT**

The agent cannot pitch the product until it has asked qualifying questions about the prospect's needs, budget, and situation.

## R02 — Demo Before Negotiate

> **Must OFFER_DEMO before NEGOTIATE**

No discount or price negotiation is allowed unless a product demo has been offered and scheduled.

## R03 — Budget Known Before Negotiate

> **Budget must be known before NEGOTIATE**

The prospect's budget must be revealed (via QUALIFY action) before the agent can enter negotiations.

## R04 — Discount After Objections

> **Discount only after 2 objections handled**

If the agent mentions a discount during NEGOTIATE, at least 2 prospect objections must have been successfully handled first.

## R05 — No Repeat Action

> **Cannot repeat same action consecutively**

The agent cannot use the same action type twice in a row. A QUALIFY cannot follow a QUALIFY, a PRESENT cannot follow a PRESENT, etc.

## R06 — First Action Must Be PROSPECT

> **First action must always be PROSPECT**

Every episode must begin with the PROSPECT action. Any other first action is invalid.

## R07 — Follow-Up Only After Silence

> **FOLLOW_UP only after prospect goes silent**

FOLLOW_UP is only valid when the prospect has disengaged (returned a `silence` response). If the prospect just responded with actual content, FOLLOW_UP is a violation.

## R08 — Disqualify Logic

> **DISQUALIFY only if prospect is genuinely unqualified**

DISQUALIFY is a violation if the prospect is actually closable (true budget ≥ close threshold AND decision maker is present). Use it only when the deal is truly unwinnable.

## R09 — Close Requires Demo

> **Must OFFER_DEMO before CLOSE (difficulty 2+)**

On difficulty 2 and above, the agent must have completed OFFER_DEMO before attempting to CLOSE the deal.

---

## How Rules Are Enforced

Rules are checked **before** the prospect responds to an action. Violations are accumulated in `constraints_violated` and returned in the observation:

```python
# Observation schema
{
    "constraints_violated": ["R01", "R05"],  # New violations this turn
    "steps_completed": ["PROSPECT", "QUALIFY"],
    ...
}
```

When `len(constraints_violated) >= 3`, the episode terminates with:
- `r_outcome = -0.5` (terminal penalty)
- `r_compliance = -0.2 × violations` (per-turn penalty)
