# salespath_env/server/task_bank.py

import random
from dataclasses import dataclass


@dataclass
class ProspectProfile:
    company_name: str
    company_size: str          # small / medium / enterprise
    industry: str
    budget_signal: str         # high / medium / low / unknown
    pain_points: list[str]
    decision_maker: bool

    # Hidden values — never exposed directly to agent
    true_budget: float         # 0.0 → 1.0
    close_threshold: float
    stall_probability: float


# -------------------------
# LEVEL 1 — Easy
# budget known
# decision maker present
# close is usually possible
# -------------------------

PROFILES_L1 = [
    ProspectProfile(
        company_name="Meridian Retail",
        company_size="medium",
        industry="retail",
        budget_signal="high",
        pain_points=[
            "manual inventory tracking",
            "slow reporting",
        ],
        decision_maker=True,
        true_budget=0.8,
        close_threshold=0.5,
        stall_probability=0.0,
    ),

    ProspectProfile(
        company_name="Northline Foods",
        company_size="small",
        industry="food distribution",
        budget_signal="medium",
        pain_points=[
            "supplier delays",
            "inventory mismatch",
        ],
        decision_maker=True,
        true_budget=0.6,
        close_threshold=0.5,
        stall_probability=0.0,
    ),
]


# -------------------------
# LEVEL 2 — Medium
# budget hidden initially
# one objection expected
# -------------------------

PROFILES_L2 = [
    ProspectProfile(
        company_name="Apex Logistics",
        company_size="enterprise",
        industry="logistics",
        budget_signal="unknown",
        pain_points=[
            "route optimization",
            "driver coordination",
            "fuel tracking",
        ],
        decision_maker=True,
        true_budget=0.7,
        close_threshold=0.5,
        stall_probability=0.0,
    ),

    ProspectProfile(
        company_name="Vertex Supply",
        company_size="medium",
        industry="manufacturing",
        budget_signal="unknown",
        pain_points=[
            "vendor visibility",
            "purchase delays",
        ],
        decision_maker=True,
        true_budget=0.55,
        close_threshold=0.5,
        stall_probability=0.0,
    ),
]


# -------------------------
# LEVEL 3 — Hard
# budget hidden
# 2 objections
# possible stalling
# decision maker may be absent
# -------------------------

PROFILES_L3 = [
    ProspectProfile(
        company_name="Nova Financial",
        company_size="enterprise",
        industry="finance",
        budget_signal="unknown",
        pain_points=[
            "compliance reporting",
            "audit trails",
            "data silos",
        ],
        decision_maker=False,
        true_budget=0.6,
        close_threshold=0.55,
        stall_probability=0.3,
    ),

    ProspectProfile(
        company_name="Atlas Health",
        company_size="enterprise",
        industry="healthcare",
        budget_signal="unknown",
        pain_points=[
            "patient workflow delays",
            "reporting compliance",
        ],
        decision_maker=False,
        true_budget=0.65,
        close_threshold=0.55,
        stall_probability=0.25,
    ),
]


# -------------------------
# LEVEL 4 — Trap cases
# misleading signals
# correct action may be DISQUALIFY
# -------------------------

PROFILES_L4 = [
    ProspectProfile(
        company_name="Cipher Tech",
        company_size="small",
        industry="technology",
        budget_signal="high",   # misleading
        pain_points=[
            "security",
            "compliance",
        ],
        decision_maker=True,
        true_budget=0.2,
        close_threshold=0.5,
        stall_probability=0.5,
    ),

    ProspectProfile(
        company_name="BluePeak Studio",
        company_size="small",
        industry="creative agency",
        budget_signal="high",   # misleading
        pain_points=[
            "project visibility",
            "client reporting",
        ],
        decision_maker=True,
        true_budget=0.25,
        close_threshold=0.5,
        stall_probability=0.4,
    ),
]


ALL_PROFILES = {
    1: PROFILES_L1,
    2: PROFILES_L2,
    3: PROFILES_L3,
    4: PROFILES_L4,
}


def sample_profile(difficulty: int) -> ProspectProfile:
    """
    Returns one sampled profile for the selected difficulty.
    """

    if difficulty not in ALL_PROFILES:
        difficulty = 1

    return random.choice(ALL_PROFILES[difficulty])