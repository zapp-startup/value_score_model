"""Synthetic data for small-scale training and testing."""

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