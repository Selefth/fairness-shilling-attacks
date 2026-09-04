"""Build the paper's budget-curve figures of the C-Fair attacks.

Each figure is a 3-by-N groupplot: one column per attack, one row per dataset,
plotting the benefit ratio gamma against the attack budget for a family of
models.

Three things vary independently, and each is a registry rather than a hardcoded
list, so a new combination is a dictionary entry and not a new script:

    ATTACK_SETS         which three attacks make up the columns
    FIGURES             which models make up the curves
    ATTRIBUTE_CONFIGS   which sensitive attribute, and its datasets

Two attack sets are defined, matching the two C-fair attack families:

    influencer   Influencer PushR / PushLF / NukeF
    favorite     Favorite PushR / PushLF and ReverseFavorite NukeF

and four model families, matching the four figures per set:

    neighborhood_conventional   SLIM-u, EASE-u, SLIM-i, EASE-i
    mf_conventional             SVD, MF, NMF, NeuMF
    neighborhood_fair           BNSLIM-u against its SLIM-u counterpart
    mf_fair                     the four fairness-aware MF variants against MF

Nothing here trains a model or runs an attack; these read result CSVs and emit
TikZ. Fairness-aware model names take the attribute as a suffix and the dataset
list comes from ATTRIBUTE_CONFIGS, exactly as in experiments/tables.

    python experiments/figures/attack_figures.py [--figure NAME|all]
                                                 [--attack-set NAME|all]
                                                 [--attribute gender|age|all]
                                                 [--output-dir DIR]

Output is one file per (figure, attack set, attribute), named
``<figure>_<attack_set>_<attribute>.tex``, each carrying a matching \\label so
the sets can be included side by side in the paper.

Values are taken through the run_key recorded in
each run's ndcg_index.csv, the same join experiments/tables uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.helper_functions import paths
from src.helper_functions.experiment_registry import (
    ATTRIBUTE_CONFIGS,
    DATASET_LABELS,
    DEFAULT_ATTRIBUTE,
    LIST_SIZE,
    datasets_for,
    gamma_column,
)
from src.helper_functions.result_utils import canonical_run_keys

# The three attacks that make up a figure's columns. Every model family below can
# be built for either set; the set decides the columns, the family the curves.
#
#   title         how the set is named in the caption
#   strategies    the parenthesised list the conventional captions end with
#   column_note   a sentence appended to every caption, when the columns need one
#   label_suffix  appended to the figure's \label so both sets can coexist
#   attacks       (column title, attack id), left to right
#
# Both sets title their columns by target-item strategy -- PushR, PushLF, NukeF --
# and not by attack family, for the same reason experiments/tables groups its rows
# that way: column n is then the same strategy in every figure, so two figures can
# be read against each other position by position, which is the comparison the
# paper is actually making. The family is what varies between figures, so it lives
# in the caption. It has to be spelled out for the Favorite set, whose third
# column switches to ReverseFavorite -- hence column_note.
ATTACK_SETS = {
    "influencer": {
        "title": "C-Fair Influencer Attack",
        "strategies": (
            r"(\textsc{PushR}, \textsc{PushLF}, and \textsc{NukeF})"
        ),
        "column_note": None,
        "label_suffix": "",
        "attacks": [
            (r"\textsc{PushR}", "cfair_influencer_push_random"),
            (r"\textsc{PushLF}", "cfair_influencer_push_least_favorite"),
            (r"\textsc{NukeF}", "cfair_influencer_nuke_favorite"),
        ],
    },
    "favorite": {
        "title": "C-Fair Favorite and ReverseFavorite Attacks",
        "strategies": (
            r"(\textsc{PushR}, \textsc{PushLF}, and \textsc{NukeF})"
        ),
        "column_note": (
            r"\textsc{PushR} and \textsc{PushLF} are shown in their "
            r"\textsc{Favorite} variant and \textsc{NukeF} in its "
            r"\textsc{ReverseFavorite} variant."
        ),
        "label_suffix": "_favorite",
        "attacks": [
            (r"\textsc{PushR}", "cfair_favorite_push_random"),
            (r"\textsc{PushLF}", "cfair_favorite_push_least_favorite"),
            (r"\textsc{NukeF}", "cfair_reverse_favorite_nuke_favorite"),
        ],
    },
}
DEFAULT_ATTACK_SET = "influencer"

# (pgfplots style, legend label, model name, fairness-aware?)
# A fairness-aware model carries the attribute as a suffix, so its name depends
# on which attribute the figure is being built for.
#
# Captions are templates over the attack set: {attack} becomes the set's title and
# {strategies} its parenthesised column list, so one wording serves both sets.
# CLIP_FACT is appended only when a point was actually clipped, so a caption never
# claims a clipped curve that isn't there.
FIGURES = {
    "neighborhood_conventional": {
        "models": [
            ("slimu", "SLIM-u", "slim-u", False),
            ("easeu", "EASE-u", "ease-u", False),
            ("slimi", "SLIM-i", "slim-i", False),
            ("easei", "EASE-i", "ease-i", False),
        ],
        "legend_columns": 4,
        "clip": False,
        "label": "fig:gamma_neighborhood_based_conventional_datasets",
        "short_caption": (
            "{attack} on {attribute} against neighborhood-based conventional "
            "recommenders"
        ),
        "caption": (
            "{attack} on {attribute} against neighborhood-based conventional "
            "recommenders. The benefit ratio $\\gamma$ is shown for each model "
            "across datasets, attack budgets, and attack strategies {strategies}."
        ),
    },
    "mf_conventional": {
        "models": [
            ("svd", "SVD", "svd", False),
            ("mf", "MF", "mf", False),
            ("nmf", "NMF", "nmf", False),
            ("neumf", "NeuMF", "neumf", False),
        ],
        "legend_columns": 4,
        "clip": False,
        "label": "fig:gamma_mf_based_conventional_datasets",
        "short_caption": (
            "{attack} on {attribute} against MF-based conventional recommenders"
        ),
        "caption": (
            "{attack} on {attribute} against MF-based conventional recommenders. "
            "The benefit ratio $\\gamma$ is shown for each model across datasets, "
            "attack budgets, and attack strategies {strategies}."
        ),
    },
    "neighborhood_fair": {
        "models": [
            ("slimu", "SLIM-u", "slim-u", False),
            ("bnslimu", "BNSLIM-u", "bnslim-u", True),
        ],
        "legend_columns": 2,
        "clip": True,
        "label": "fig:gamma_neighborhood_based_fair_datasets",
        "short_caption": (
            "{attack} on {attribute} against fairness-aware neighborhood-based "
            "recommender BNSLIM-u"
        ),
        "caption": (
            "{attack} on {attribute} against fairness-aware neighborhood-based "
            "recommender BNSLIM-u. The benefit ratio $\\gamma$ is shown across "
            "attack strategies, datasets, and budgets. The conventional SLIM-u "
            "counterpart is included for comparison."
        ),
    },
    "mf_fair": {
        "models": [
            ("mf", "MF", "mf", False),
            ("mfvalue", "MF-value", "mf-value", True),
            ("mfabsolute", "MF-absolute", "mf-absolute", True),
            ("mfunder", "MF-under", "mf-under", True),
            ("mfover", "MF-over", "mf-over", True),
        ],
        "legend_columns": 5,
        "clip": True,
        "label": "fig:gamma_mf_based_fair_datasets",
        "short_caption": (
            "{attack} on {attribute} against fairness-aware MF-based recommenders"
        ),
        "caption": (
            "{attack} on {attribute} against fairness-aware MF-based "
            "recommenders. The benefit ratio $\\gamma$ per model is shown across "
            "attack strategies, datasets, and attack budgets. The conventional "
            "MF counterpart is included for comparison."
        ),
    },
}

BUDGETS = [(0, 0.0), (10, 0.1), (25, 0.25), (50, 0.5), (75, 0.75), (100, 1.0)]
CLIP_AT = 1.3

# Two figures let gamma exceed 1 when both group benefits are near zero, so their
# y-axis is capped and points above it are cut off. The caption states only where
# this happens, not why -- an explanation of individual clipped points used to be
# generated here too, but it was a claim about the data that had to be verified
# against every clipped point on every rebuild, for a sentence the paper does not
# need. {clipped} is filled with the datasets it happened in.
CLIP_FACT = "For readability, values above $1.3$ are clipped (in {clipped})."


def fmt(value: float) -> str:
    return f"{value:.15g}"


def model_name(base: str, is_fair: bool, attribute: str) -> str:
    """Inverse of the runners' labelling: fairness-aware models take the suffix."""
    return f"{base}-{attribute}" if is_fair else base


def load_clean(dataset: str) -> pd.DataFrame:
    return pd.read_csv(paths.clean_results(dataset, LIST_SIZE))


def load_attack(dataset: str, attack_id: str, attribute: str) -> pd.DataFrame | None:
    """Attack rows for one dataset, restricted to the canonical NDCG checkpoint."""
    result_path = paths.attack_results(dataset, attack_id, LIST_SIZE)
    if not result_path.exists():
        return None

    attack = pd.read_csv(result_path)
    if "attribute" not in attack.columns:
        return None
    attack = attack[attack["attribute"].eq(attribute)].copy()
    if attack.empty:
        return None

    # Drop left over rows that don't match the canonical NDCG checkpoint. A
    # figure plots whatever it can, so an index that matches nothing leaves the
    # rows alone rather than emptying the curve.
    keys = canonical_run_keys(dataset, attack_id, attribute)
    if keys and "run_key" in attack.columns:
        attack = attack[attack["run_key"].isin(keys)]
    return attack


def model_values(
    clean: pd.DataFrame,
    attack: pd.DataFrame | None,
    name: str,
    attribute: str,
) -> list[tuple[int, float]] | None:
    """The gamma curve for one model, or None when any budget is missing."""
    col = gamma_column(attribute)
    clean_rows = clean[clean["model"].eq(name)]
    if clean_rows.empty or col not in clean_rows or pd.isna(clean_rows.iloc[0][col]):
        return None
    if attack is None or attack.empty:
        return None

    values = [(0, float(clean_rows.iloc[0][col]))]
    for budget_percent, budget_csv in BUDGETS[1:]:
        rows = attack[
            attack["model"].eq(name)
            & np.isclose(attack["budget"].astype(float), budget_csv)
        ]
        if rows.empty or pd.isna(rows.iloc[0][col]):
            return None
        if len(rows) > 1 and rows[col].nunique() > 1:
            raise ValueError(
                f"Ambiguous rows for model={name} budget={budget_csv} "
                f"attribute={attribute}: {sorted(rows[col].unique())}. "
                "The canonical-row join did not reduce this to one value."
            )
        values.append((budget_percent, float(rows.iloc[0][col])))
    return values


def addplot(style: str, values: list[tuple[int, float]]) -> list[str]:
    lines = [f"\\addplot[{style}] table {{"]
    lines.extend(f"{budget} {fmt(value)}" for budget, value in values)
    lines.append("};")
    return lines


def groupplot_options(
    row_label: str | None,
    title: str | None,
    xlabel: bool,
    legend_columns: int | None,
) -> str:
    opts = []
    if title is not None:
        opts.append(f"    title={{{title}}}")
    if row_label is not None:
        opts.append(f"    ylabel={{{row_label}}}")
    if xlabel:
        opts.append("    xlabel={Budget (\\%)}")
    opts.extend(
        [
            "    extra y ticks={0.8}",
            "    extra y tick style={grid=major, grid style={black, dashed}}",
            "    axis on top",
        ]
    )
    if not xlabel:
        opts.append("    xticklabels={}")
    if legend_columns is not None:
        opts.extend(
            [
                f"    legend columns={legend_columns}",
                "    legend cell align=left",
                (
                    "    legend style={\n"
                    "        font=\\footnotesize,\n"
                    "        at={(0.5,-0.55)},\n"
                    "        anchor=north,\n"
                    "        draw=none,\n"
                    "        column sep=0.08cm,\n"
                    "        inner xsep=1pt,\n"
                    "        inner ysep=1pt,\n"
                    "        nodes={inner sep=0pt, outer sep=0pt},\n"
                    "    }"
                ),
            ]
        )
    return ",\n".join(opts)


def caption_text(template: str, attack_set: str, attribute: str) -> str:
    """Fill the attack-set and attribute placeholders without str.format, which
    would choke on the braces every other token in a LaTeX caption is made of."""
    spec = ATTACK_SETS[attack_set]
    return (
        template.replace("{attack}", spec["title"])
        .replace("{strategies}", spec["strategies"])
        .replace("{attribute}", attribute)
    )


def figure_header(datasets: list[str], clip: bool) -> list[str]:
    """The groupplot preamble: geometry, ticks, and the optional y clip."""
    header = [
        r"\begin{figure*}",
        r"\centering",
        r"\begin{tikzpicture}",
        r"[scale=0.95, transform shape]",
        r"\begin{groupplot}[",
        r"    group style={",
        f"        group size=3 by {len(datasets)},",
        r"        horizontal sep=1.0cm,",
        r"        vertical sep=0.3cm,",
        r"    },",
        r"    width=5.0cm,",
        r"    height=3.7cm,",
        r"    xtick={0,10,25,50,75,100},",
        r"    xticklabels={0,10,25,50,75,100},",
        r"    xticklabel style={font=\footnotesize},",
        r"    yticklabel style={font=\footnotesize},",
        f"    ymin=0, ymax={CLIP_AT:g},",
    ]
    if clip:
        header.append(f"    restrict y to domain=0:{CLIP_AT:g},")
    header.append("]")
    return header


def clipped_points(
    values: list[tuple[int, float]], dataset: str, name: str
) -> list[dict]:
    """Which points on this curve landed above the clip, for the report line."""
    return [
        {"dataset": dataset, "model": name, "budget": budget, "gamma": gamma}
        for budget, gamma in values
        if gamma > CLIP_AT
    ]


def figure_caption(
    spec: dict, attacks: dict, attribute: str, clipped: list[dict]
) -> tuple[str, str, str]:
    """Caption, short caption and label for one figure."""
    caption = caption_text(spec["caption"], attack_set=attacks["name"], attribute=attribute)
    if attacks["column_note"]:
        caption += " " + attacks["column_note"]
    if spec["clip"] and clipped:
        where = " and ".join(dict.fromkeys(p["dataset_label"] for p in clipped))
        caption += "\n" + CLIP_FACT.replace("{clipped}", where)
    label = spec["label"] + attacks["label_suffix"]
    if attribute != DEFAULT_ATTRIBUTE:
        label += f"_{attribute}"
    short_caption = caption_text(spec["short_caption"], attack_set=attacks["name"], attribute=attribute)
    return caption, short_caption, label


def generate(
    figure_name: str,
    attribute: str,
    attack_set: str = DEFAULT_ATTACK_SET,
) -> tuple[str, dict]:
    """One figure: a groupplot of gamma against budget, plus what the run found."""
    spec = FIGURES[figure_name]
    attacks = {**ATTACK_SETS[attack_set], "name": attack_set}
    datasets = datasets_for(attribute)
    models = [
        (style, label, model_name(base, is_fair, attribute))
        for style, label, base, is_fair in spec["models"]
    ]

    lines = figure_header(datasets, spec["clip"])
    stats = {"curves": 0, "empty_cells": 0, "clipped": [], "missing": []}

    for row_idx, dataset in enumerate(datasets):
        label = DATASET_LABELS[dataset]
        clean = load_clean(dataset)
        is_last_row = row_idx == len(datasets) - 1
        lines.append(
            f"% ===================== Row {row_idx + 1}: {label} ====================="
        )
        for col_idx, (attack_title, attack_id) in enumerate(attacks["attacks"]):
            attack = load_attack(dataset, attack_id, attribute)
            legend = spec["legend_columns"] if (is_last_row and col_idx == 1) else None
            lines.append(r"\nextgroupplot[")
            lines.append(
                groupplot_options(
                    row_label=label if col_idx == 0 else None,
                    title=attack_title if row_idx == 0 else None,
                    xlabel=is_last_row,
                    legend_columns=legend,
                )
            )
            lines.append("]")

            plotted = False
            for style, _legend, name in models:
                values = model_values(clean, attack, name, attribute)
                if values is None:
                    stats["missing"].append(f"{dataset}/{attack_id}/{name}")
                    continue
                for point in clipped_points(values, dataset, name):
                    stats["clipped"].append({**point, "dataset_label": label})
                lines.extend(addplot(style, values))
                stats["curves"] += 1
                plotted = True

            if legend is not None:
                for style, legend_label, _name in models:
                    lines.append(f"\\addlegendimage{{{style}}}")
                    lines.append(f"\\addlegendentry{{{legend_label}}}")

            if not plotted:
                stats["empty_cells"] += 1
                lines.append(
                    f"% Empty: no complete {attribute} rows for this "
                    "dataset/attack in the CSVs."
                )

    caption, short_caption, label = figure_caption(
        spec, attacks, attribute, stats["clipped"]
    )
    lines.extend(
        [
            r"\end{groupplot}",
            r"\end{tikzpicture}",
            f"\\caption[{short_caption}]{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{figure*}",
            "",
        ]
    )
    return "\n".join(lines), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        choices=sorted(FIGURES) + ["all"],
        default="all",
        help="Which figure family to build.",
    )
    parser.add_argument(
        "--attack-set",
        choices=sorted(ATTACK_SETS) + ["all"],
        default=DEFAULT_ATTACK_SET,
        help="Which three attacks form the columns. 'all' builds every set.",
    )
    parser.add_argument(
        "--attribute",
        choices=sorted(ATTRIBUTE_CONFIGS) + ["all"],
        default=DEFAULT_ATTRIBUTE,
        help="Sensitive attribute. 'all' builds every attribute.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.figures_dir(),
        help="Root for the generated .tex files; <attribute>/<attack set>/ below it.",
    )
    args = parser.parse_args()

    figures = sorted(FIGURES) if args.figure == "all" else [args.figure]
    attack_sets = (
        sorted(ATTACK_SETS) if args.attack_set == "all" else [args.attack_set]
    )
    attributes = (
        sorted(ATTRIBUTE_CONFIGS) if args.attribute == "all" else [args.attribute]
    )
    for attribute in attributes:
        for attack_set in attack_sets:
            for figure_name in figures:
                tex, stats = generate(figure_name, attribute, attack_set)
                path = args.output_dir / attribute / attack_set / f"{figure_name}.tex"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(tex)
                report(path, stats, attribute)


def report(path, stats: dict, attribute: str) -> None:
    """One line per figure, plus anything the run could not take for granted."""
    clipped = stats["clipped"]
    note = f"{stats['curves']} curves"
    if stats["empty_cells"]:
        note += f", {stats['empty_cells']} empty cells"
    if clipped:
        where = ", ".join(dict.fromkeys(p["dataset_label"] for p in clipped))
        note += f", {len(clipped)} points above {CLIP_AT:g} (clipped; {where})"
    print(f"wrote {path}  [{note}]")

    # Never drop a curve silently: a missing model is a data gap worth seeing.
    for miss in stats["missing"]:
        print(f"    no complete curve: {miss}")


if __name__ == "__main__":
    main()
