"""
Synthetic data for training and evaluation.

``generate_sample_dataset`` — default generator for small-scale runs.
``generate_eval_harness_dataset`` — diverse personas and configurable history for stress tests.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
import random


def generate_sample_dataset(
    n_users: int = 50,
    n_merchants: int = 20,
    transactions_per_user: tuple[int, int] = (3, 80),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames:
      merchants, subscriptions, subscription_usage, transactions,
      user_explicit, user_computed, user_inferred, user_facts, feedback_signals
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    now = datetime.utcnow()

    # ── Merchants ────────────────────────────────────────────────
    categories = ["streaming", "grocery", "fitness", "software",
                  "utilities", "food", "education", "other"]
    merchant_names = [
        "Netflix", "Spotify", "Amazon Prime", "Hulu", "Disney+",
        "Adobe CC", "GitHub", "Notion", "Gym Membership", "Costco",
        "Headspace", "Duolingo Plus", "Dropbox", "Microsoft 365",
        "NYT Digital", "PlayStation Plus", "Crunchyroll", "Uber Eats",
        "Audible", "YouTube Premium"
    ][:n_merchants]

    merchants = pd.DataFrame({
        "id": range(1, n_merchants + 1),
        "name": merchant_names,
        "category": rng.choice(categories, n_merchants),
    })

    # ── User Explicit ────────────────────────────────────────────
    life_stages = ["student", "early_career", "mid_career", "family", "other"]
    financial_goals = ["save_more", "invest", "reduce_debt", "control_subs", "other"]
    risk_tolerances = ["low", "medium", "high"]
    budget_styles = ["strict", "flexible", "optimize_value"]

    # value sliders — randomized but they'll be normalized later
    cost_sliders = rng.integers(20, 90, n_users)
    quality_sliders = rng.integers(20, 90, n_users)
    sust_sliders = rng.integers(10, 70, n_users)

    user_explicit = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "financial_goal": rng.choice(financial_goals, n_users),
        "risk_tolerance": rng.choice(risk_tolerances, n_users),
        "budget_style": rng.choice(budget_styles, n_users),
        "life_stage": rng.choice(life_stages, n_users),
        "monthly_income": rng.uniform(2000, 12000, n_users).round(2),
        "monthly_fixed_expenses": rng.uniform(500, 3000, n_users).round(2),
        "value_priority_cost": cost_sliders,
        "value_priority_quality": quality_sliders,
        "value_priority_sustainability": sust_sliders,
    })

    # Normalize value priority sliders to sum to 1 → stored in user_computed
    total = user_explicit[["value_priority_cost", "value_priority_quality",
                            "value_priority_sustainability"]].sum(axis=1)
    cost_w = (user_explicit["value_priority_cost"] / total).round(4)
    qual_w = (user_explicit["value_priority_quality"] / total).round(4)
    sust_w = (user_explicit["value_priority_sustainability"] / total).round(4)

    # ── User Computed ────────────────────────────────────────────
    user_computed = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "cost_weight": cost_w.values,
        "quality_weight": qual_w.values,
        "sustainability_weight": sust_w.values,
        "impulse_susceptibility_score": rng.uniform(0.1, 0.9, n_users).round(3),
        "regret_sensitivity": rng.uniform(0.1, 0.8, n_users).round(3),
        "budget_adherence_score": rng.uniform(0.2, 1.0, n_users).round(3),
        "spending_personality": rng.choice(["Saver", "Spender", "Balanced"], n_users),
        "product_spending_style": rng.choice(["Planned", "Mixed", "Impulsive"], n_users),
        "subscription_behavior_type": rng.choice(
            ["Loyal", "Churn-heavy", "Over-subscribed", "Minimal"], n_users
        ),
    })

    # ── User Inferred ────────────────────────────────────────────
    user_inferred = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "avg_purchase_price": rng.uniform(5, 80, n_users).round(2),
        "percent_impulsive_purchases": rng.uniform(0.0, 0.6, n_users).round(3),
        "regret_frequency": rng.uniform(0.0, 0.5, n_users).round(3),
        "active_subscriptions_count": rng.integers(1, 10, n_users),
        "total_subscription_cost": rng.uniform(20, 300, n_users).round(2),
        "percent_income_spent_on_subscriptions": rng.uniform(0.01, 0.15, n_users).round(4),
        "cancel_reactivation_frequency": rng.uniform(0.0, 0.4, n_users).round(3),
        "actual_monthly_spending": rng.uniform(800, 5000, n_users).round(2),
    })

    # ── Subscriptions ────────────────────────────────────────────
    sub_rows = []
    sub_id = 1
    billing_cycles = ["monthly", "yearly", "weekly"]
    for user_id in range(1, n_users + 1):
        n_subs = rng.integers(1, min(6, n_merchants))
        chosen_merchants = rng.choice(merchants["id"].values, n_subs, replace=False)
        for mid in chosen_merchants:
            cat = merchants.loc[merchants["id"] == mid, "category"].values[0]
            base_price = {"streaming": 15, "fitness": 40, "software": 20,
                          "education": 10, "grocery": 50, "utilities": 60,
                          "food": 30, "other": 12}.get(cat, 12)
            started = now - timedelta(days=int(rng.integers(30, 730)))
            sub_rows.append({
                "id": sub_id,
                "user_id": user_id,
                "merchant_id": int(mid),
                "price": round(base_price * rng.uniform(0.7, 1.5), 2),
                "billing_cycle": rng.choice(billing_cycles, p=[0.7, 0.2, 0.1]),
                "status": rng.choice(["active", "paused", "canceled"], p=[0.8, 0.1, 0.1]),
                "started_on": started.date(),
                "renewal_date": (started + timedelta(days=30)).date(),
                "usage_frequency": float(rng.integers(1, 14)),
                "reactivation_count": int(rng.integers(0, 4)),
            })
            sub_id += 1

    subscriptions = pd.DataFrame(sub_rows)

    # ── Subscription usage (raw metrics for preprocessor → generic signals) ──
    usage_rows = []
    for _, sub in subscriptions.iterrows():
        mid = int(sub["merchant_id"])
        name = str(merchants.loc[merchants["id"] == mid, "name"].iloc[0]).lower()
        row: dict = {"subscription_id": int(sub["id"])}
        if "netflix" in name:
            row["hours_watched"] = float(rng.uniform(5, 80))
        elif "spotify" in name:
            row["minutes_listened"] = float(rng.uniform(100, 2500))
        elif "amazon prime" in name:
            row["delivery_savings_usd"] = float(rng.uniform(0, 35))
        elif "uber eats" in name or "ubereats" in name.replace(" ", ""):
            row["food_savings_usd"] = float(rng.uniform(0, 40))
        usage_rows.append(row)
    subscription_usage = pd.DataFrame(usage_rows)

    # ── Transactions ─────────────────────────────────────────────
    txn_rows = []
    txn_id = 1
    for _, sub in subscriptions.iterrows():
        n_txns = int(rng.integers(*transactions_per_user))
        user_comp = user_computed[user_computed["user_id"] == sub["user_id"]].iloc[0]

        for _ in range(n_txns):
            days_ago = int(rng.integers(0, 365))
            occurred = now - timedelta(days=days_ago)

            # Simulate correlated satisfaction: better for lower regret users
            regret_sens = user_comp["regret_sensitivity"]
            sat = int(np.clip(rng.normal(7 - regret_sens * 3, 1.5), 1, 10))
            regret = int(np.clip(rng.normal(regret_sens * 60, 15), 0, 100))
            repurchase = int(np.clip(rng.normal((sat / 10) * 80, 15), 0, 100))

            # Sparse feedback — not every transaction has it
            has_feedback = rng.random() < 0.6
            has_text = rng.random() < 0.3

            txn_rows.append({
                "id": txn_id,
                "user_id": int(sub["user_id"]),
                "merchant_id": int(sub["merchant_id"]),
                "subscription_id": int(sub["id"]),
                "amount": sub["price"],
                "occurred_at": occurred,
                "direction": "spend",
                "category": "subscriptions",
                "satisfaction_rating": sat if has_feedback else None,
                "regret_rating": regret if has_feedback else None,
                "repurchase_likelihood": repurchase if has_feedback else None,
                "usage_frequency": float(rng.integers(1, 14)) if has_feedback else None,
                "reflection_text": _sample_reflection(sat, rng) if has_text else None,
                "impulse_score": round(float(rng.uniform(0, 0.4)), 3),
                "regret_score": round(regret_sens * float(rng.uniform(0.5, 1.0)), 3),
                "used_buy_advisor": bool(rng.random() < 0.2),
            })
            txn_id += 1

    transactions = pd.DataFrame(txn_rows)

    # ── User Facts ───────────────────────────────────────────────
    fact_keys = ["values_convenience", "values_quality", "values_price_sensitivity",
                 "prefers_annual_billing", "avoids_impulsive_spend"]
    fact_rows = []
    for user_id in range(1, n_users + 1):
        n_facts = rng.integers(1, 4)
        for fk in rng.choice(fact_keys, n_facts, replace=False):
            fact_rows.append({
                "user_id": user_id,
                "fact_key": fk,
                "fact_value": round(float(rng.uniform(0.4, 1.0)), 3),
                "confidence": round(float(rng.uniform(0.5, 1.0)), 3),
            })
    user_facts = pd.DataFrame(fact_rows)

    # ── Feedback Signals (simulated future model output) ─────────
    # Only exists for a subset of (user, merchant) pairs
    fb_rows = []
    for _, sub in subscriptions.sample(frac=0.3, random_state=seed).iterrows():
        fb_rows.append({
            "user_id": int(sub["user_id"]),
            "merchant_id": int(sub["merchant_id"]),
            "subscription_id": int(sub["id"]),
            "transaction_id": None,
            "feedback_value_score": round(float(rng.uniform(0.2, 0.95)), 3),
            "feedback_confidence": round(float(rng.uniform(0.3, 0.95)), 3),
            "scored_at": now,
        })
    feedback_signals = pd.DataFrame(fb_rows)

    return {
        "merchants": merchants,
        "subscriptions": subscriptions,
        "subscription_usage": subscription_usage,
        "transactions": transactions,
        "user_explicit": user_explicit,
        "user_computed": user_computed,
        "user_inferred": user_inferred,
        "user_facts": user_facts,
        "feedback_signals": feedback_signals,
    }


PERSONAS = [
    "balanced",
    "thin_file",
    "heavy_user",
    "stable_saver",
    "volatile_spender",
    "low_income_stressed",
    "high_income_prime",
    "essentials_heavy",
    "discretionary_heavy",
    "subscription_maxed",
    "edge_no_feedback",
]


def generate_eval_harness_dataset(
    n_users: int = 2000,
    n_merchants: int = 48,
    months_history: int = 6,
    seed: int = 42,
    reference_time: datetime | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Synthetic dataset tuned for value-score stress tests: diverse personas,
    configurable history window, and a companion profile table for exports.

    Returns
    -------
    data : dict of DataFrames
        Same keys as ``generate_sample_dataset`` (merchants, subscriptions,
        transactions, user_*, feedback_signals).
    profile : DataFrame
        One row per ``user_id`` with generation metadata (persona, spend
        proxies, category-mix notes). Columns not consumed by the model
        pipeline — use only for evaluation exports.

    Notes
    -----
    Credit utilization / overdraft / late-payment signals are **not** modeled
    in ``FeatureEngineer``; ``credit_stress_proxy`` and ``overdraft_risk_proxy``
    in ``profile`` are best-effort labels for offline analysis only.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    ref = reference_time or datetime.utcnow()
    horizon_days = max(30, int(months_history) * 30)

    weights = np.array(
        [0.22, 0.08, 0.12, 0.10, 0.08, 0.10, 0.08, 0.08, 0.08, 0.08, 0.06],
        dtype=float,
    )
    weights /= weights.sum()

    categories = [
        "streaming",
        "grocery",
        "fitness",
        "software",
        "utilities",
        "food",
        "education",
        "other",
    ]
    n_merchants = min(n_merchants, max(8, len(categories) * 6))
    per_cat = max(1, n_merchants // len(categories))
    merchant_cats = (categories * (per_cat + 1))[:n_merchants]
    rng.shuffle(merchant_cats)

    merchants = pd.DataFrame(
        {
            "id": range(1, n_merchants + 1),
            "name": [f"Merchant_{i}" for i in range(1, n_merchants + 1)],
            "category": merchant_cats,
        }
    )

    personas = rng.choice(PERSONAS, size=n_users, p=weights)

    life_stages = [
        "student",
        "early_career",
        "mid_career",
        "family",
        "pre_retirement",
        "other",
    ]
    financial_goals = [
        "save_more",
        "invest",
        "reduce_debt",
        "build_credit",
        "control_subs",
        "other",
    ]
    risk_tolerances = ["low", "medium", "high"]
    budget_styles = ["strict", "flexible", "optimize_value"]

    monthly_income = np.zeros(n_users)
    fixed_exp = np.zeros(n_users)
    cost_sliders = np.zeros(n_users, dtype=int)
    qual_sliders = np.zeros(n_users, dtype=int)
    sust_sliders = np.zeros(n_users, dtype=int)
    credit_stress = np.zeros(n_users)
    overdraft_risk = np.zeros(n_users)
    n_subs_target = np.zeros(n_users, dtype=int)
    txn_mult = np.ones(n_users)

    for i, p in enumerate(personas):
        if p == "low_income_stressed":
            monthly_income[i] = rng.uniform(900, 3200)
            fixed_exp[i] = rng.uniform(400, min(2200, monthly_income[i] * 0.85))
            cost_sliders[i] = int(rng.integers(55, 92))
            qual_sliders[i] = int(rng.integers(15, 45))
            sust_sliders[i] = int(rng.integers(8, 35))
            credit_stress[i] = rng.uniform(0.45, 0.95)
            overdraft_risk[i] = rng.uniform(0.35, 0.9)
            n_subs_target[i] = int(rng.integers(2, 5))
            txn_mult[i] = rng.uniform(0.7, 1.2)
        elif p == "high_income_prime":
            monthly_income[i] = rng.uniform(9500, 32000)
            fixed_exp[i] = rng.uniform(2500, 9000)
            cost_sliders[i] = int(rng.integers(18, 55))
            qual_sliders[i] = int(rng.integers(35, 85))
            sust_sliders[i] = int(rng.integers(20, 65))
            credit_stress[i] = rng.uniform(0.02, 0.25)
            overdraft_risk[i] = rng.uniform(0.0, 0.15)
            n_subs_target[i] = int(rng.integers(3, 8))
            txn_mult[i] = rng.uniform(0.9, 1.6)
        elif p == "thin_file":
            monthly_income[i] = rng.uniform(2000, 5200)
            fixed_exp[i] = rng.uniform(700, 2400)
            cost_sliders[i] = int(rng.integers(30, 70))
            qual_sliders[i] = int(rng.integers(25, 70))
            sust_sliders[i] = int(rng.integers(15, 50))
            credit_stress[i] = rng.uniform(0.15, 0.55)
            overdraft_risk[i] = rng.uniform(0.1, 0.45)
            n_subs_target[i] = int(rng.integers(1, 3))
            txn_mult[i] = rng.uniform(0.25, 0.55)
        elif p == "heavy_user":
            monthly_income[i] = rng.uniform(3500, 14000)
            fixed_exp[i] = rng.uniform(1200, 4500)
            cost_sliders[i] = int(rng.integers(25, 65))
            qual_sliders[i] = int(rng.integers(25, 75))
            sust_sliders[i] = int(rng.integers(15, 55))
            credit_stress[i] = rng.uniform(0.1, 0.45)
            overdraft_risk[i] = rng.uniform(0.05, 0.35)
            n_subs_target[i] = int(rng.integers(4, 8))
            txn_mult[i] = rng.uniform(1.6, 2.4)
        elif p == "stable_saver":
            monthly_income[i] = rng.uniform(4500, 11000)
            fixed_exp[i] = rng.uniform(1800, 4200)
            cost_sliders[i] = int(rng.integers(45, 85))
            qual_sliders[i] = int(rng.integers(20, 55))
            sust_sliders[i] = int(rng.integers(25, 60))
            credit_stress[i] = rng.uniform(0.05, 0.3)
            overdraft_risk[i] = rng.uniform(0.0, 0.2)
            n_subs_target[i] = int(rng.integers(2, 5))
            txn_mult[i] = rng.uniform(0.75, 1.1)
        elif p == "volatile_spender":
            monthly_income[i] = rng.uniform(2800, 9000)
            fixed_exp[i] = rng.uniform(800, 3800)
            cost_sliders[i] = int(rng.integers(20, 60))
            qual_sliders[i] = int(rng.integers(30, 80))
            sust_sliders[i] = int(rng.integers(10, 45))
            credit_stress[i] = rng.uniform(0.25, 0.7)
            overdraft_risk[i] = rng.uniform(0.15, 0.55)
            n_subs_target[i] = int(rng.integers(3, 7))
            txn_mult[i] = rng.uniform(0.9, 1.8)
        elif p == "essentials_heavy":
            monthly_income[i] = rng.uniform(3200, 9000)
            fixed_exp[i] = rng.uniform(1400, 4000)
            cost_sliders[i] = int(rng.integers(40, 75))
            qual_sliders[i] = int(rng.integers(20, 50))
            sust_sliders[i] = int(rng.integers(15, 45))
            credit_stress[i] = rng.uniform(0.15, 0.5)
            overdraft_risk[i] = rng.uniform(0.1, 0.4)
            n_subs_target[i] = int(rng.integers(3, 7))
            txn_mult[i] = rng.uniform(1.0, 1.5)
        elif p == "discretionary_heavy":
            monthly_income[i] = rng.uniform(4000, 12000)
            fixed_exp[i] = rng.uniform(1500, 4000)
            cost_sliders[i] = int(rng.integers(15, 45))
            qual_sliders[i] = int(rng.integers(40, 85))
            sust_sliders[i] = int(rng.integers(15, 50))
            credit_stress[i] = rng.uniform(0.1, 0.45)
            overdraft_risk[i] = rng.uniform(0.05, 0.35)
            n_subs_target[i] = int(rng.integers(4, 9))
            txn_mult[i] = rng.uniform(1.2, 2.0)
        elif p == "subscription_maxed":
            monthly_income[i] = rng.uniform(3500, 10000)
            fixed_exp[i] = rng.uniform(1200, 4000)
            cost_sliders[i] = int(rng.integers(25, 60))
            qual_sliders[i] = int(rng.integers(25, 65))
            sust_sliders[i] = int(rng.integers(15, 45))
            credit_stress[i] = rng.uniform(0.2, 0.6)
            overdraft_risk[i] = rng.uniform(0.1, 0.45)
            n_subs_target[i] = int(rng.integers(7, min(12, n_merchants)))
            txn_mult[i] = rng.uniform(0.85, 1.35)
        elif p == "edge_no_feedback":
            monthly_income[i] = rng.uniform(2500, 7500)
            fixed_exp[i] = rng.uniform(900, 3200)
            cost_sliders[i] = int(rng.integers(25, 75))
            qual_sliders[i] = int(rng.integers(25, 75))
            sust_sliders[i] = int(rng.integers(15, 50))
            credit_stress[i] = rng.uniform(0.1, 0.5)
            overdraft_risk[i] = rng.uniform(0.05, 0.4)
            n_subs_target[i] = int(rng.integers(2, 5))
            txn_mult[i] = rng.uniform(0.8, 1.2)
        else:  # balanced
            monthly_income[i] = rng.uniform(2800, 11000)
            fixed_exp[i] = rng.uniform(900, 3800)
            cost_sliders[i] = int(rng.integers(25, 75))
            qual_sliders[i] = int(rng.integers(25, 75))
            sust_sliders[i] = int(rng.integers(15, 55))
            credit_stress[i] = rng.uniform(0.1, 0.45)
            overdraft_risk[i] = rng.uniform(0.05, 0.35)
            n_subs_target[i] = int(rng.integers(2, 6))
            txn_mult[i] = rng.uniform(0.85, 1.25)

    monthly_income = np.round(monthly_income, 2)
    fixed_exp = np.round(fixed_exp, 2)

    user_explicit = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "financial_goal": rng.choice(financial_goals, n_users),
            "risk_tolerance": rng.choice(risk_tolerances, n_users),
            "budget_style": rng.choice(budget_styles, n_users),
            "life_stage": rng.choice(life_stages, n_users),
            "monthly_income": monthly_income,
            "monthly_fixed_expenses": fixed_exp,
            "value_priority_cost": cost_sliders,
            "value_priority_quality": qual_sliders,
            "value_priority_sustainability": sust_sliders,
        }
    )

    total = user_explicit[
        ["value_priority_cost", "value_priority_quality", "value_priority_sustainability"]
    ].sum(axis=1)
    cost_w = (user_explicit["value_priority_cost"] / total).round(4)
    qual_w = (user_explicit["value_priority_quality"] / total).round(4)
    sust_w = (user_explicit["value_priority_sustainability"] / total).round(4)

    impulse = np.zeros(n_users)
    regret_sens = np.zeros(n_users)
    budget_adh = np.zeros(n_users)
    for i, p in enumerate(personas):
        if p == "stable_saver":
            impulse[i] = rng.uniform(0.05, 0.35)
            regret_sens[i] = rng.uniform(0.35, 0.65)
            budget_adh[i] = rng.uniform(0.75, 1.0)
        elif p in ("volatile_spender", "discretionary_heavy"):
            impulse[i] = rng.uniform(0.45, 0.95)
            regret_sens[i] = rng.uniform(0.45, 0.85)
            budget_adh[i] = rng.uniform(0.25, 0.65)
        elif p == "low_income_stressed":
            impulse[i] = rng.uniform(0.2, 0.65)
            regret_sens[i] = rng.uniform(0.45, 0.8)
            budget_adh[i] = rng.uniform(0.3, 0.7)
        else:
            impulse[i] = rng.uniform(0.1, 0.75)
            regret_sens[i] = rng.uniform(0.2, 0.75)
            budget_adh[i] = rng.uniform(0.35, 0.95)

    user_computed = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "cost_weight": cost_w.values,
            "quality_weight": qual_w.values,
            "sustainability_weight": sust_w.values,
            "impulse_susceptibility_score": np.round(impulse, 3),
            "regret_sensitivity": np.round(regret_sens, 3),
            "budget_adherence_score": np.round(budget_adh, 3),
            "spending_personality": rng.choice(["Saver", "Spender", "Balanced"], n_users),
            "product_spending_style": rng.choice(["Planned", "Mixed", "Impulsive"], n_users),
            "subscription_behavior_type": rng.choice(
                ["Loyal", "Churn-heavy", "Over-subscribed", "Minimal"],
                n_users,
            ),
        }
    )

    pct_sub = np.clip(
        rng.normal(0.08, 0.04, n_users) * (monthly_income / 5000).clip(0.5, 2.0),
        0.01,
        0.22,
    )
    actual_spend = np.clip(
        monthly_income * rng.uniform(0.55, 1.15, n_users) * txn_mult,
        400,
        monthly_income * 1.4,
    )

    user_inferred = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "avg_purchase_price": rng.uniform(8, 95, n_users).round(2),
            "percent_impulsive_purchases": rng.uniform(0.0, 0.55, n_users).round(3),
            "regret_frequency": rng.uniform(0.0, 0.45, n_users).round(3),
            "active_subscriptions_count": np.clip(n_subs_target + rng.integers(-1, 2, n_users), 1, 15),
            "total_subscription_cost": rng.uniform(25, 420, n_users).round(2),
            "percent_income_spent_on_subscriptions": pct_sub.round(4),
            "cancel_reactivation_frequency": rng.uniform(0.0, 0.45, n_users).round(3),
            "actual_monthly_spending": actual_spend.round(2),
        }
    )

    essential_cats = {"grocery", "utilities", "education"}
    disc_cats = {"streaming", "fitness", "software", "food", "other"}

    sub_rows = []
    sub_id = 1
    billing_cycles = ["monthly", "yearly", "weekly"]

    for user_id in range(1, n_users + 1):
        p = personas[user_id - 1]
        n_subs = int(n_subs_target[user_id - 1])
        n_subs = max(1, min(n_subs, n_merchants))

        if p == "essentials_heavy":
            pool = merchants[merchants["category"].isin(essential_cats)]["id"].values
            if len(pool) < n_subs:
                pool = merchants["id"].values
        elif p == "discretionary_heavy":
            pool = merchants[merchants["category"].isin(disc_cats)]["id"].values
            if len(pool) < n_subs:
                pool = merchants["id"].values
        else:
            pool = merchants["id"].values

        chosen_merchants = rng.choice(pool, size=min(n_subs, len(pool)), replace=False)
        if len(chosen_merchants) < n_subs:
            extra = rng.choice(merchants["id"].values, n_subs - len(chosen_merchants), replace=False)
            chosen_merchants = np.unique(np.concatenate([chosen_merchants, extra]))[:n_subs]

        for mid in chosen_merchants:
            mid = int(mid)
            cat = merchants.loc[merchants["id"] == mid, "category"].values[0]
            base_price = {
                "streaming": 15,
                "fitness": 42,
                "software": 22,
                "education": 12,
                "grocery": 55,
                "utilities": 95,
                "food": 28,
                "other": 14,
            }.get(cat, 14)
            started = ref - timedelta(days=int(rng.integers(20, horizon_days)))
            status = "active"
            if rng.random() < 0.06 and p != "thin_file":
                status = rng.choice(["paused", "canceled"])
            sub_rows.append(
                {
                    "id": sub_id,
                    "user_id": user_id,
                    "merchant_id": mid,
                    "price": round(base_price * float(rng.uniform(0.65, 1.55)), 2),
                    "billing_cycle": rng.choice(billing_cycles, p=[0.72, 0.2, 0.08]),
                    "status": status,
                    "started_on": started.date(),
                    "renewal_date": (started + timedelta(days=30)).date(),
                    "usage_frequency": float(rng.integers(1, 14)),
                    "reactivation_count": int(rng.integers(0, 5)),
                }
            )
            sub_id += 1

    subscriptions = pd.DataFrame(sub_rows)

    for uid in range(1, n_users + 1):
        sub_uid = subscriptions[subscriptions["user_id"] == uid]
        if not sub_uid["status"].eq("active").any():
            ix = sub_uid.index[0]
            subscriptions.loc[ix, "status"] = "active"

    txn_rows = []
    txn_id = 1
    for _, sub in subscriptions.iterrows():
        uid = int(sub["user_id"])
        p = personas[uid - 1]
        cat = merchants.loc[merchants["id"] == sub["merchant_id"], "category"].values[0]

        if p == "thin_file":
            low, high = 2, 5
        elif p == "heavy_user":
            low, high = 75, min(horizon_days - 1, 165)
        elif p == "edge_no_feedback":
            low, high = 10, 35
        else:
            low, high = 12, 48
        low = max(2, int(low * txn_mult[uid - 1]))
        high = max(low + 1, int(high * txn_mult[uid - 1]))
        high = min(high, horizon_days)
        low = min(low, high - 1)
        n_txns = int(rng.integers(low, high + 1))

        days_ago = rng.choice(np.arange(1, horizon_days + 1), size=n_txns, replace=False)
        user_comp = user_computed[user_computed["user_id"] == uid].iloc[0]
        regret_sens = float(user_comp["regret_sensitivity"])

        for d in days_ago:
            occurred = ref - timedelta(days=int(d))
            if p == "edge_no_feedback":
                has_feedback = False
            else:
                has_feedback = rng.random() < 0.62
            has_text = has_feedback and (rng.random() < 0.28)

            if has_feedback:
                sat = int(np.clip(rng.normal(7 - regret_sens * 3, 1.6), 1, 10))
                regret = int(np.clip(rng.normal(regret_sens * 58, 16), 0, 100))
                repurchase = int(np.clip(rng.normal((sat / 10) * 78, 16), 0, 100))
            else:
                sat = regret = repurchase = None

            impulse_jitter = 0.15 if p == "volatile_spender" else 0.0
            txn_rows.append(
                {
                    "id": txn_id,
                    "user_id": uid,
                    "merchant_id": int(sub["merchant_id"]),
                    "subscription_id": int(sub["id"]),
                    "amount": round(float(sub["price"]) * float(rng.uniform(0.92, 1.08)), 2),
                    "occurred_at": occurred,
                    "direction": "spend",
                    "category": str(cat),
                    "satisfaction_rating": sat if has_feedback else None,
                    "regret_rating": regret if has_feedback else None,
                    "repurchase_likelihood": repurchase if has_feedback else None,
                    "usage_frequency": float(rng.integers(1, 14)) if has_feedback else None,
                    "reflection_text": _sample_reflection(sat if sat else 5, rng) if has_text else None,
                    "impulse_score": round(
                        float(rng.uniform(0, 0.35 + impulse_jitter)), 3
                    ),
                    "regret_score": round(regret_sens * float(rng.uniform(0.45, 1.0)), 3),
                    "used_buy_advisor": bool(rng.random() < 0.22),
                }
            )
            txn_id += 1

    transactions = pd.DataFrame(txn_rows)

    fact_keys = [
        "values_convenience",
        "values_quality",
        "values_price_sensitivity",
        "prefers_annual_billing",
        "avoids_impulsive_spend",
    ]
    fact_rows = []
    for user_id in range(1, n_users + 1):
        n_facts = int(rng.integers(1, 4))
        for fk in rng.choice(fact_keys, n_facts, replace=False):
            fact_rows.append(
                {
                    "user_id": user_id,
                    "fact_key": fk,
                    "fact_value": round(float(rng.uniform(0.35, 1.0)), 3),
                    "confidence": round(float(rng.uniform(0.55, 1.0)), 3),
                }
            )
    user_facts = pd.DataFrame(fact_rows)

    fb_rows = []
    fb_frac = 0.35 if len(subscriptions) > 0 else 0.0
    for _, sub in subscriptions.sample(frac=fb_frac, random_state=seed).iterrows():
        fb_rows.append(
            {
                "user_id": int(sub["user_id"]),
                "merchant_id": int(sub["merchant_id"]),
                "subscription_id": int(sub["id"]),
                "transaction_id": None,
                "feedback_value_score": round(float(rng.uniform(0.15, 0.95)), 3),
                "feedback_confidence": round(float(rng.uniform(0.35, 0.95)), 3),
                "scored_at": ref,
            }
        )
    feedback_signals = pd.DataFrame(fb_rows)

    mix_rows = []
    for user_id in range(1, n_users + 1):
        ut = transactions[transactions["user_id"] == user_id]
        if ut.empty:
            mix_rows.append(
                {
                    "user_id": user_id,
                    "txn_count_horizon": 0,
                    "avg_monthly_txn_count": 0.0,
                    "category_diversity": 0,
                    "essentials_txn_share": 0.0,
                    "discretionary_txn_share": 0.0,
                    "transfer_style_proxy": False,
                    "cash_withdrawal_proxy": False,
                }
            )
            continue
        by_cat = ut.groupby(ut["category"])["id"].count()
        n_tx = len(ut)
        ess = sum(by_cat.get(c, 0) for c in essential_cats)
        disc = sum(by_cat.get(c, 0) for c in disc_cats)
        mix_rows.append(
            {
                "user_id": user_id,
                "txn_count_horizon": int(n_tx),
                "avg_monthly_txn_count": round(n_tx / max(months_history, 1), 4),
                "category_diversity": int(ut["category"].nunique()),
                "essentials_txn_share": round(ess / n_tx, 4),
                "discretionary_txn_share": round(disc / n_tx, 4),
                "transfer_style_proxy": bool(personas[user_id - 1] == "volatile_spender" and rng.random() < 0.12),
                "cash_withdrawal_proxy": bool(
                    personas[user_id - 1] in ("low_income_stressed", "volatile_spender")
                    and rng.random() < 0.08
                ),
            }
        )
    mix_df = pd.DataFrame(mix_rows)

    profile = pd.DataFrame(
        {
            "user_id": range(1, n_users + 1),
            "persona": personas,
            "months_history": months_history,
            "credit_stress_proxy": np.round(credit_stress, 4),
            "overdraft_risk_proxy": np.round(overdraft_risk, 4),
        }
    )
    profile = profile.merge(mix_df, on="user_id", how="left")

    data = {
        "merchants": merchants,
        "subscriptions": subscriptions,
        "transactions": transactions,
        "user_explicit": user_explicit,
        "user_computed": user_computed,
        "user_inferred": user_inferred,
        "user_facts": user_facts,
        "feedback_signals": feedback_signals,
    }
    return data, profile


def _sample_reflection(sat_rating: int, rng: np.random.Generator) -> str:
    positive = [
        "Really happy with this, use it almost every day.",
        "Great value for the price, would definitely keep it.",
        "Been using this for months and it never disappoints.",
        "Totally worth it, improved my daily routine.",
    ]
    neutral = [
        "It's okay, use it sometimes but not as much as I expected.",
        "Decent but I'm not sure if I'll renew.",
        "Use it occasionally. Might reconsider at renewal.",
    ]
    negative = [
        "Barely use this anymore, probably going to cancel.",
        "Regret getting this, doesn't match what I needed.",
        "Not worth the price for how little I use it.",
    ]
    if sat_rating >= 7:
        pool = positive
    elif sat_rating >= 4:
        pool = neutral
    else:
        pool = negative
    return rng.choice(pool)