"""Two measurements quoted in the RQ1 discussion that the attack tables cannot show.

Multiple-testing family size
----------------------------
Each (dataset, model, attack) combination contributes one Mann--Whitney test of
whether the unprivileged group lost more benefit than the privileged group. The
vanilla table contains 195 such tests and the C-fair table 390, so a raw count
of p < 0.05 says little: at alpha = 0.05 roughly ten of the 195 vanilla tests
would clear that threshold even if no attack had any effect. Both tables
therefore report significance after a Benjamini--Hochberg correction.

How much that correction inflates a p-value depends on how many tests are
corrected together, and each table corrects its own attack family in isolation.
Treating all 585 tests as a single family is strictly more conservative, and so
bounds how far the reported contrast between the two attack families depends on
where that boundary is drawn. ``pooled_bh_check`` reports both corrections side
by side, together with the split of the vanilla results between conventional and
fairness-aware models.

Group composition of the copied profiles
-----------------------------------------
PowerUser and the C-Fair Influencer attack both build fake profiles by copying
highly active real users, and differ only in the pool they draw from: PowerUser
uses the whole population, C-Fair Influencer only the unprivileged group. That
makes PowerUser the baseline closest to our attack, and the comparison between
them informative only insofar as the two pools actually differ -- which depends
on how activity is distributed across groups in a given dataset. Were the most
active users of a dataset predominantly unprivileged, PowerUser would
approximate the C-Fair Influencer attack and the two would not be separable.

``poweruser_group_composition`` measures the unprivileged share among the users
PowerUser copies on ML1M, the dataset where the vanilla baselines have their
largest effect on C-fairness. The share falls below the population share, so the
copied behaviour is characteristic of the privileged group. Since the injected
profiles are nevertheless labelled unprivileged, as fairness-aware training
requires a group for every user, the label rather than the injected behaviour
accounts for the effect these baselines have on fairness-aware models.

Run from the repository root:

    .venv/bin/python experiments/tables/rq1_contrast_check.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from experiments.tables import attack_tables
from src.helper_functions import paths
from src.helper_functions.experiment_registry import (
    ALPHA,
    CFAIR_ATTACKS,
    DEFAULT_ATTRIBUTE,
    KIND_LABELS,
    KINDS,
    VANILLA_ATTACKS,
    groups_config,
)
from src.helper_functions.model_naming import (
    get_base_model_for,
    strip_attribute_suffix,
)
from src.helper_functions.result_utils import (
    benjamini_hochberg,
    budget_to_fake_users,
    infer_unprivileged_privileged,
)

ATTRIBUTE = DEFAULT_ATTRIBUTE


def pooled_bh_check(budget: float, attribute: str = ATTRIBUTE) -> str:
    """Compare the per-family and pooled Benjamini--Hochberg corrections.

    The tables adjust the 195 vanilla and 390 C-fair p-values as two separate
    families. Pooling all 585 into one family raises every adjusted p-value, so
    whatever remains significant under the pooled correction is significant
    under the per-family one as well.
    """
    vanilla = attack_tables.load_significance(
        VANILLA_ATTACKS, attack_tables.load_vanilla_rows, budget, attribute
    )
    vanilla["family"] = "vanilla"
    cfair = attack_tables.load_significance(
        CFAIR_ATTACKS, attack_tables.load_cfair_rows, budget, attribute
    )
    cfair["family"] = "C-fair"

    pooled = pd.concat([vanilla, cfair], ignore_index=True)
    pooled["p_value_bh_pooled"] = benjamini_hochberg(pooled["p_value"])
    pooled["significant_bh_pooled"] = pooled["p_value_bh_pooled"] < ALPHA

    lines = [
        "Benjamini--Hochberg family size",
        f"  all model-wise tests at a {100 * budget:.0f}% budget: {len(pooled)}",
        "",
        f"  {'family':<8} {'tests':>6} {'uncorr.':>8} {'per-family':>11} {'pooled':>8}",
    ]
    for family in ["vanilla", "C-fair"]:
        sub = pooled[pooled["family"].eq(family)]
        lines.append(
            f"  {family:<8} {len(sub):>6} {int(sub['significant_uncorrected'].sum()):>8} "
            f"{int(sub['significant_bh'].sum()):>11} "
            f"{int(sub['significant_bh_pooled'].sum()):>8}"
        )
    vanilla_pooled_min = pooled.loc[
        pooled["family"].eq("vanilla"), "p_value_bh_pooled"
    ].min()
    lines += [
        "",
        f"  smallest pooled adjusted p among vanilla tests: {vanilla_pooled_min:.3f}",
        "  The contrast between the two attack families does not depend on",
        "  correcting them separately.",
    ]

    hits = vanilla[vanilla["significant_uncorrected"]]
    totals = vanilla.groupby("model_kind").size()
    by_kind = hits.groupby("model_kind").size()
    lines += ["", "  vanilla tests reaching p < 0.05 before correction:"]
    for kind in KINDS:
        lines.append(
            f"    {KIND_LABELS[kind]:<15} "
            f"{int(by_kind.get(kind, 0))}/{int(totals.get(kind, 0))}"
        )
    return "\n".join(lines)


def unprivileged_group(dataset: str, attribute: str = ATTRIBUTE) -> str | None:
    """The unprivileged group of a dataset, when its models agree on one.

    Derived per model with ``infer_unprivileged_privileged``, the function the
    experiment runners use. Fairness-aware models take the split of their
    conventional counterpart via ``model_naming.get_base_model_for``, again as
    the runners do, so a model whose intervention has reversed the disparity does
    not flip the group. Returns ``None`` when the models do not agree, in which
    case a single share over the copied profiles has no meaning.
    """
    clean = attack_tables.load_clean(dataset, attribute).set_index("model")
    groups = set()
    for model in clean.index:
        reference = model
        if attack_tables.model_kind(model, attribute) == "fair":
            reference = get_base_model_for(strip_attribute_suffix(model, attribute))
        group, _ = infer_unprivileged_privileged(
            clean.loc[reference], attribute, groups_config(attribute)
        )
        groups.add(group)
    return groups.pop() if len(groups) == 1 else None


def poweruser_group_composition(budget: float) -> str:
    """Measure the unprivileged share among the profiles PowerUser copies on ML1M.

    PowerUser selects the most active users in the population; the C-Fair
    Influencer attack selects the most active users within the unprivileged
    group. This quantifies how far apart those two selections fall on ML1M,
    the dataset where the vanilla baselines have their largest effect on
    C-fairness.

    Requires the raw MovieLens-1M files under ``data/ml-1m``.
    """
    ratings_path = paths.data_dir("ml-1m") / "ratings.dat"
    users_path = paths.data_dir("ml-1m") / "users.dat"
    if not ratings_path.exists() or not users_path.exists():
        return (
            "PowerUser profile composition on ML1M\n"
            "  skipped: raw data/ml-1m files not found."
        )

    ratings = pd.read_csv(
        ratings_path, sep="::", engine="python", names=["user", "item", "rating", "ts"]
    )
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user", "gender", "age", "occupation", "zip"],
    )
    activity = (
        ratings.groupby("user").size().rename("n").reset_index().merge(users, on="user")
    )
    activity = activity.sort_values("n", ascending=False)

    group = unprivileged_group("ml-1m")
    if group is None:
        return (
            "PowerUser profile composition on ML1M\n"
            "  skipped: the models do not agree on a single unprivileged group."
        )
    unprivileged = group.upper()
    population_share = float((activity["gender"] == unprivileged).mean())
    n_fake = budget_to_fake_users(budget, int((activity["gender"] == unprivileged).sum()))
    copied_share = float((activity.head(n_fake)["gender"] == unprivileged).mean())

    return (
        "PowerUser profile composition on ML1M\n"
        f"  unprivileged group, agreed by all 13 models: {unprivileged}\n"
        f"  fake users injected at a {100 * budget:.0f}% budget: {n_fake}\n"
        f"  unprivileged share among the copied profiles: {100 * copied_share:.1f}%\n"
        f"  unprivileged share in the population:         "
        f"{100 * population_share:.1f}%\n"
        "  The copied profiles under-represent the unprivileged group, so the\n"
        "  injected behaviour is characteristic of the privileged group. The\n"
        "  injected users still carry the unprivileged label that fairness-aware\n"
        "  training assigns them."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budget", type=float, default=0.1, help="Attack budget to measure."
    )
    args = parser.parse_args()

    print(pooled_bh_check(args.budget))
    print()
    print(poweruser_group_composition(args.budget))


if __name__ == "__main__":
    main()
