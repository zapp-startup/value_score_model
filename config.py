import yaml
from pathlib import Path
def load_config(path=None) -> dict:

    if path is None:
        path = Path(__file__).parent / "configs" / "default.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def scoring_scale(config: dict) -> dict[str, float]:
    """Learned (tier) cap and display (base + bonus) cap from config."""
    s = config.get("scoring", {})
    learned = float(s.get("learned_score_max", s.get("base_score_max", 100)))
    display = float(s.get("display_max", s.get("max_score", 150)))
    return {"learned_max": learned, "display_max": display}


def recommendation_thresholds(config: dict) -> tuple[str, dict[str, float]]:
    """
    Returns (score_column, thresholds) where score_column is 'combined' or 'base'
    and thresholds keys: strong_buy, buy, skip on the appropriate 0–100 or 0–display scale.
    """
    r = config.get("recommendation", {})
    col = str(r.get("score_column", "combined")).lower()
    if col == "base":
        return col, {
            "strong_buy": float(r.get("base_strong_buy_threshold", 85)),
            "buy": float(r.get("base_buy_threshold", 65)),
            "skip": float(r.get("base_skip_threshold", 35)),
        }
    return "combined", {
        "strong_buy": float(r.get("strong_buy_threshold", 127.5)),
        "buy": float(r.get("buy_threshold", 97.5)),
        "skip": float(r.get("skip_threshold", 52.5)),
    }