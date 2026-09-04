"""What the experiment is made of, in one place.

``paths`` owns where results live; this owns what they are called and which
combinations exist. 
"""

from __future__ import annotations

import re

LIST_SIZE = 10
TARGET_ITEMS = 10

# Four-fifths rule: a benefit ratio below this counts as disparate impact.
GAMMA_THRESHOLD = 0.8
ALPHA = 0.05

# Every dataset, in the order tables and figures present them.
DATASETS = ["ml-100k", "ml-1m", "lastfm-1k", "ftky", "fnyc"]
DATASET_LABELS = {
    "ml-100k": "ML100K",
    "ml-1m": "ML1M",
    "lastfm-1k": "LFM1K",
    "ftky": "FTKY",
    "fnyc": "FNYC",
}

# One entry per sensitive attribute the analysis can be built for.
#   groups   -- the two group codes, as the per-group result columns name them
#   datasets -- those actually run for this attribute
#   families -- the attack families that have results for it
ATTRIBUTE_CONFIGS = {
    "gender": {
        "groups": ["f", "m"],
        "datasets": ["ml-100k", "ml-1m", "lastfm-1k", "ftky", "fnyc"],
        "families": ["vanilla", "cfair"],
    },
    "age": {
        "groups": ["y", "o"],
        "datasets": ["ml-100k", "ml-1m", "lastfm-1k"],
        "families": ["cfair"],
    },
}
DEFAULT_ATTRIBUTE = "gender"

# Attack ids and the names the paper gives them.
VANILLA_ATTACKS = [
    ("power_user", r"\textsc{PowerUser}"),
    ("bandwagon", r"\textsc{Bandwagon}"),
    ("reverse_bandwagon", r"\textsc{ReverseBandwagon}"),
]
CFAIR_ATTACKS = [
    ("cfair_influencer_push_random", r"\textsc{Influencer-PushR}"),
    ("cfair_influencer_push_least_favorite", r"\textsc{Influencer-PushLF}"),
    ("cfair_influencer_nuke_favorite", r"\textsc{Influencer-NukeF}"),
    ("cfair_favorite_push_random", r"\textsc{Favorite-PushR}"),
    ("cfair_favorite_push_least_favorite", r"\textsc{Favorite-PushLF}"),
    ("cfair_reverse_favorite_nuke_favorite", r"\textsc{ReverseFavorite-NukeF}"),
]

KINDS = ["base", "fair"]
KIND_LABELS = {"base": "Conventional", "fair": "Fairness-aware"}


def check_attribute(attribute: str) -> str:
    if attribute not in ATTRIBUTE_CONFIGS:
        raise ValueError(
            f"Unknown attribute {attribute!r}; "
            f"expected one of {sorted(ATTRIBUTE_CONFIGS)}"
        )
    return attribute


def groups_for(attribute: str) -> list[str]:
    """The two group codes, unprivileged first only by convention of the data."""
    return list(ATTRIBUTE_CONFIGS[check_attribute(attribute)]["groups"])


def groups_config(attribute: str) -> dict[str, list[str]]:
    """The shape ``result_utils`` expects: {attribute: [group, group]}."""
    return {attribute: groups_for(attribute)}


def datasets_for(attribute: str) -> list[str]:
    """The datasets this attribute was run on."""
    return list(ATTRIBUTE_CONFIGS[check_attribute(attribute)]["datasets"])


def families_for(attribute: str) -> list[str]:
    """The attack families with results for this attribute."""
    return list(ATTRIBUTE_CONFIGS[check_attribute(attribute)]["families"])


def other_attribute_suffixes(attribute: str) -> tuple[str, ...]:
    """Name endings that mark a model as belonging to a different attribute.

    A dataset's clean results hold the fairness-aware models of every attribute
    that was run, so these are what tell ``mf-value-age`` apart from a
    conventional model when the gender tables are being built.
    """
    return tuple(f"-{a}" for a in ATTRIBUTE_CONFIGS if a != attribute)


def gamma_column(attribute: str, metric: str = "ndcg") -> str:
    """The benefit-ratio column, as ``result_utils.add_group_ratios`` names it."""
    return f"{metric}_{attribute}_ratio"


def plain_label(label: str) -> str:
    """Drop the LaTeX wrapper from a label, for plain-text output."""
    return re.sub(r"\\textsc\{([^}]*)\}", r"\1", label)
