"""Interactive exploration of attack results, backing 04_plots.ipynb.

Companion to attack_figures.py, which builds the fixed paper figures. This one is
for looking around: any dataset, attribute, attack, model and metric, plotted on
screen. It writes nothing.

Two ideas keep it usable:

* Everything is discovered from the result files. ``options()`` prints what is
  actually there.
* Every argument except the dataset has a sensible default. ``plot_attack("ml-1m")``
  works; narrow it down only when you want to.

    from experiments.figures import explore
    explore.options()                       # what exists, everywhere
    explore.options("ml-1m")                # what exists for one dataset
    explore.plot_attack("ml-1m")            # every model, default attack
    explore.plot_attack("ml-1m", attack_id="power_user", models=["svd", "mf"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.helper_functions import paths
from src.helper_functions.model_naming import get_base_model_for, strip_attribute_suffix
from src.helper_functions.result_utils import infer_unprivileged_privileged
from experiments.figures import attack_figures

LIST_SIZE = attack_figures.LIST_SIZE
GROUPS_CONFIG = {
    attribute: config["groups"]
    for attribute, config in attack_figures.ATTRIBUTE_CONFIGS.items()
}
METRICS = ["ndcg", "precision", "recall"]


# ---------------------------------------------------------------------------
# discovery -- read from the files, never from a hardcoded list
# ---------------------------------------------------------------------------


def datasets() -> list[str]:
    return [d for d in attack_figures.DATASET_LABELS if paths.dataset_dir(d).is_dir()]


def attributes(dataset: str) -> list[str]:
    """Attributes this dataset actually has per-group columns for."""
    columns = set(pd.read_csv(paths.clean_results(dataset, LIST_SIZE)).columns)
    return [
        attribute
        for attribute, (g1, g2) in GROUPS_CONFIG.items()
        if f"ndcg_{g1}" in columns and f"ndcg_{g2}" in columns
    ]


def attacks(dataset: str, attribute: str | None = None) -> list[str]:
    """Attacks with results for this dataset, optionally for one attribute."""
    found = paths.available_attacks(dataset, LIST_SIZE)
    if attribute is None:
        return found
    keep = []
    for attack_id in found:
        rows = attack_figures.load_attack(dataset, attack_id, attribute)
        if rows is not None and not rows.empty:
            keep.append(attack_id)
    return keep


def models(dataset: str, attribute: str | None = None) -> list[str]:
    """Models in this dataset's clean results, excluding other attributes' variants."""
    names = list(pd.read_csv(paths.clean_results(dataset, LIST_SIZE))["model"])
    if attribute is None:
        return sorted(names)
    others = tuple(f"-{a}" for a in GROUPS_CONFIG if a != attribute)
    return sorted(n for n in names if not (others and n.endswith(others)))


def budgets(dataset: str, attack_id: str, attribute: str) -> list[float]:
    """Budgets present for this combination, with 0 (the clean model) first."""
    rows = attack_figures.load_attack(dataset, attack_id, attribute)
    if rows is None or rows.empty:
        return [0.0]
    return [0.0] + sorted(float(b) for b in rows["budget"].unique())


def options(dataset: str | None = None, attribute: str | None = None) -> None:
    """Print what is available, read live from results/."""
    targets = datasets() if dataset is None else [dataset]
    for ds in targets:
        attrs = attributes(ds)
        print(f"{ds}  ({attack_figures.DATASET_LABELS.get(ds, ds)})")
        print(f"  attributes: {', '.join(attrs) or 'none'}")
        for attr in attrs if attribute is None else [attribute]:
            if attr not in attrs:
                print(f"  {attr}: not available for this dataset")
                continue
            available = attacks(ds, attr)
            print(f"  {attr}:")
            print(f"    attacks ({len(available)}): {', '.join(available) or 'none'}")
            print(f"    models  ({len(models(ds, attr))}): {', '.join(models(ds, attr))}")
            if available:
                print(f"    budgets: {budgets(ds, available[0], attr)}")
        print(f"  metrics: {', '.join(METRICS)}")
        print()


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def _group_split(clean: pd.DataFrame, model: str, attribute: str) -> tuple[str, str]:
    """The split the runners recorded: a fairness-aware model borrows its
    conventional counterpart's, so an attack targets the same users either way."""
    reference = get_base_model_for(strip_attribute_suffix(model, attribute))
    rows = clean[clean["model"].eq(reference)]
    if rows.empty:
        rows = clean[clean["model"].eq(model)]
    return infer_unprivileged_privileged(rows.iloc[0], attribute, GROUPS_CONFIG)


def plot_attack(
    dataset: str,
    attribute: str | None = None,
    attack_id: str | None = None,
    models_to_plot: list[str] | None = None,
    budgets_to_plot: list[float] | None = None,
    metric: str = "ndcg",
    ax_height: float = 4.0,
):
    """Four panels against budget: gamma, the benefit gap, and each group's score.

    Everything but ``dataset`` defaults to what the results contain. Returns the
    matplotlib axes so a caller can restyle before showing.
    """
    import matplotlib.pyplot as plt

    available_attributes = attributes(dataset)
    if not available_attributes:
        raise ValueError(f"{dataset} has no per-group columns; nothing to plot.")
    attribute = attribute or available_attributes[0]
    if attribute not in available_attributes:
        raise ValueError(
            f"{dataset} has no {attribute!r} results. Available: {available_attributes}"
        )

    available_attacks = attacks(dataset, attribute)
    if not available_attacks:
        raise ValueError(f"No attack results for {dataset}/{attribute}.")
    attack_id = attack_id or available_attacks[0]
    if attack_id not in available_attacks:
        raise ValueError(
            f"{attack_id!r} has no {attribute} results for {dataset}. "
            f"Available: {available_attacks}"
        )

    chosen_models = models_to_plot or models(dataset, attribute)
    chosen_budgets = budgets_to_plot or budgets(dataset, attack_id, attribute)

    clean = pd.read_csv(paths.clean_results(dataset, LIST_SIZE))
    attack = attack_figures.load_attack(dataset, attack_id, attribute)
    ratio_col = f"{metric}_{attribute}_ratio"
    ticks = [b * 100 for b in chosen_budgets]

    _, axes = plt.subplots(1, 4, figsize=(5 * ax_height, ax_height))
    plotted = 0
    for model in chosen_models:
        clean_rows = clean[clean["model"].eq(model)]
        if clean_rows.empty:
            print(f"skipping {model}: not in the clean results")
            continue
        unpriv, priv = _group_split(clean, model, attribute)
        series = {"gamma": [], "gap": [], "unpriv": [], "priv": []}
        incomplete = False
        for budget in chosen_budgets:
            if budget == 0:
                row = clean_rows.iloc[0]
            else:
                match = attack[
                    attack["model"].eq(model)
                    & np.isclose(attack["budget"].astype(float), budget)
                ]
                if match.empty:
                    incomplete = True
                    break
                row = match.iloc[0]
            series["gamma"].append(row[ratio_col])
            series["unpriv"].append(row[f"{metric}_{unpriv}"] * 100)
            series["priv"].append(row[f"{metric}_{priv}"] * 100)
            series["gap"].append(
                abs(row[f"{metric}_{unpriv}"] - row[f"{metric}_{priv}"]) * 100
            )
        if incomplete:
            print(f"skipping {model}: missing at least one budget")
            continue
        for ax, key in zip(axes, ["gamma", "gap", "unpriv", "priv"]):
            ax.plot(ticks, series[key], label=model, marker="o", markersize=3)
        plotted += 1

    if not plotted:
        raise ValueError("Nothing plotted; no model had a complete curve.")

    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xlabel("Budget (%)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$\gamma$")
    axes[1].set_ylabel(rf"$\Delta m$ ({metric}, %)")
    axes[2].set_ylabel(r"unprivileged $m$ (%)")
    axes[3].set_ylabel(r"privileged $m$ (%)")
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    # Limits follow the data: a fairness intervention that overshoots puts gamma
    # above 1, and a fixed ceiling would silently cut those curves off.
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize="small")
    axes[0].set_title(f"{dataset} / {attribute} / {attack_id}", loc="left", fontsize="small")
    plt.tight_layout()
    return axes
