"""Build the LaTeX tables of shilling attacks on C-fairness.

Two tables are produced from the same pipeline. The vanilla table covers
PowerUser, Bandwagon and ReverseBandwagon; the C-fair table covers the six
C-fairness shilling attacks. Both report, for the conventional and the
fairness-aware models separately, the change in the benefit ratio, the largest
benefit gap in percentage points, the count below the four-fifths threshold, and
the count of significant reductions after a Benjamini--Hochberg correction
applied across the model-wise tests of that table.

The number of model-wise tests depends on the attribute, because age labels
exist for fewer datasets than gender: the C-fair table holds 390 tests for
gender (5 datasets) but 234 for age (3 datasets), and the vanilla table 195 for
gender. That count is therefore never hardcoded here. It is computed from the
significance frame and written into the ``\\caption`` of the emitted ``.tex``
file, so a table and its stated correction scope cannot drift apart.

The attack results are produced elsewhere; this script reads and formats them.

    python experiments/tables/attack_tables.py [--family vanilla|cfair|both]
                                               [--attribute gender|age|all]
                                               [--budget 0.1] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.helper_functions import paths
from src.helper_functions.experiment_registry import (
    ALPHA,
    CFAIR_ATTACKS,
    DATASET_LABELS,
    DEFAULT_ATTRIBUTE,
    GAMMA_THRESHOLD,
    KIND_LABELS,
    KINDS,
    LIST_SIZE,
    TARGET_ITEMS,
    VANILLA_ATTACKS,
    ATTRIBUTE_CONFIGS,
    datasets_for,
    families_for,
    gamma_column,
    groups_config,
    other_attribute_suffixes,
    plain_label,
)
from src.helper_functions.model_naming import model_kind_from_result_name
from src.helper_functions.result_utils import (
    benjamini_hochberg,
    group_benefit_gap,
    infer_unprivileged_privileged,
    load_clean_results,
    select_canonical_rows,
)

# (dataset, attack_id, attribute, budget) -> the 13 rows of that combination.
RowLoader = Callable[[str, str, str, float], pd.DataFrame]

# Target-item strategy (Sec. 4.2) -> the selected-item strategies (Sec. 4.3)
# combined with it, in display order. Promoting targets first, then demoting.
CFAIR_TARGET_GROUPS = [
    (
        r"\textsc{PushR}",
        [
            ("cfair_influencer_push_random", r"\textsc{Influencer}"),
            ("cfair_favorite_push_random", r"\textsc{Favorite}"),
        ],
    ),
    (
        r"\textsc{PushLF}",
        [
            ("cfair_influencer_push_least_favorite", r"\textsc{Influencer}"),
            ("cfair_favorite_push_least_favorite", r"\textsc{Favorite}"),
        ],
    ),
    (
        r"\textsc{NukeF}",
        [
            ("cfair_influencer_nuke_favorite", r"\textsc{Influencer}"),
            ("cfair_reverse_favorite_nuke_favorite", r"\textsc{ReverseFavorite}"),
        ],
    ),
]
# The index of the first demoting group, which gets a rule above it.
CFAIR_FIRST_NUKE_GROUP = 2


# --------------------------------------------------------------------------
# loading and validation
# --------------------------------------------------------------------------


def required_columns(attribute: str, family: str) -> list[str]:
    """Columns a result row must carry for this attribute to be tabled."""
    group_a, group_b = groups_config(attribute)[attribute]
    shared = [
        "model",
        "attribute",
        "target_items",
        "budget",
        "unprivileged_group",
        "privileged_group",
        f"ndcg_{group_a}",
        f"ndcg_{group_b}",
        gamma_column(attribute),
        "p_value",
    ]
    # C-fair rows are selected through the NDCG checkpoint, so they must carry
    # the key that join goes through.
    return ["run_key", *shared] if family == "cfair" else shared


def model_kind(model: str, attribute: str) -> str:
    """Classify a model name as conventional or fairness-aware.

    Inverts ``model_naming.model_result_name``, which labels a fairness-aware
    model by appending the protected attribute to its name. Both directions of
    the convention live in ``model_naming`` so they cannot drift apart.
    """
    return model_kind_from_result_name(model, attribute)


def expected_models(clean: pd.DataFrame, attribute: str) -> set[str]:
    """The eight conventional and five <attribute>-aware model names.

    A dataset's clean results hold the fairness-aware models of every attribute
    that was run, so the ones belonging to other attributes are dropped here
    rather than counted as conventional.
    """
    others = other_attribute_suffixes(attribute)
    models = {m for m in clean["model"] if not (others and m.endswith(others))}
    kinds = (
        pd.Series(list(models))
        .map(lambda m: model_kind(m, attribute))
        .value_counts()
        .to_dict()
    )
    if kinds != {"base": 8, "fair": 5}:
        raise ValueError(
            f"Expected 8 conventional and 5 {attribute}-aware clean models, "
            f"found {kinds}: {sorted(models)}"
        )
    return models


@lru_cache(maxsize=None)
def _load_clean_validated(dataset: str, attribute: str) -> pd.DataFrame:
    """Read and validate one dataset's pre-attack results, once per run.

    Every loader call validates against this frame and every table row merges
    against it, so without the cache the same five files are read and revalidated
    dozens of times per run. Callers get a copy, so the cached frame cannot be
    mutated from outside. The attribute is part of the key because it decides
    which models belong in the frame and which gamma column is validated.
    """
    path = paths.dataset_dir(dataset)
    gamma_col = gamma_column(attribute)
    clean = load_clean_results(path, LIST_SIZE)
    clean = clean[clean["model"].isin(expected_models(clean, attribute))].copy()
    if clean["model"].duplicated().any():
        duplicates = sorted(
            clean.loc[clean["model"].duplicated(False), "model"].unique()
        )
        raise ValueError(f"Duplicate clean rows in {path}: {duplicates}")
    if clean[gamma_col].isna().any():
        missing = sorted(clean.loc[clean[gamma_col].isna(), "model"].tolist())
        raise ValueError(f"Missing clean {gamma_col} values in {path}: {missing}")
    return clean


def load_clean(dataset: str, attribute: str) -> pd.DataFrame:
    """Pre-attack results for one dataset, restricted to the 13 tabled models."""
    return _load_clean_validated(dataset, attribute).copy()


def validate_attack_rows(
    attack: pd.DataFrame, dataset: str, source: Path, attribute: str
) -> pd.DataFrame:
    expected = expected_models(load_clean(dataset, attribute), attribute)
    found = set(attack["model"])
    duplicate_models = sorted(
        attack.loc[attack["model"].duplicated(False), "model"].unique()
    )
    missing_models = sorted(expected - found)
    unexpected_models = sorted(found - expected)
    if len(attack) != 13 or duplicate_models or missing_models or unexpected_models:
        raise ValueError(
            f"Expected one row for each of 13 models in {source}; rows={len(attack)}, "
            f"duplicates={duplicate_models}, missing={missing_models}, "
            f"unexpected={unexpected_models}"
        )
    return attack


def _select_budget(frame: pd.DataFrame, attribute: str, budget: float) -> pd.DataFrame:
    """The rows of one attribute, target-item count and budget."""
    return frame[
        frame["attribute"].eq(attribute)
        & frame["target_items"].eq(TARGET_ITEMS)
        & np.isclose(frame["budget"].astype(float), budget)
    ].copy()


def _check_complete(attack: pd.DataFrame, columns: list[str], source: Path) -> None:
    incomplete = attack[columns].isna().any(axis=1)
    if incomplete.any():
        details = attack.loc[incomplete, columns].to_dict("records")
        raise ValueError(f"Incomplete rows in {source}: {details}")


def load_vanilla_rows(
    dataset: str, attack_id: str, attribute: str, budget: float
) -> pd.DataFrame:
    path = paths.attack_results(dataset, attack_id, LIST_SIZE)
    attack = pd.read_csv(path)
    columns = required_columns(attribute, "vanilla")
    missing_columns = sorted(set(columns) - set(attack.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    attack = validate_attack_rows(
        _select_budget(attack, attribute, budget), dataset, path, attribute
    )
    _check_complete(attack, columns, path)
    return attack


def load_cfair_rows(
    dataset: str, attack_id: str, attribute: str, budget: float
) -> pd.DataFrame:
    """Load the rows referenced by the current per-user NDCG checkpoint.
    """
    path = paths.attack_results(dataset, attack_id, LIST_SIZE)
    attack = pd.read_csv(path)
    columns = required_columns(attribute, "cfair")
    missing_columns = sorted(set(columns) - set(attack.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {missing_columns}")

    attack = select_canonical_rows(
        _select_budget(attack, attribute, budget),
        dataset,
        attack_id,
        attribute,
        target_items=TARGET_ITEMS,
        budget=budget,
    ).copy()
    attack = validate_attack_rows(attack, dataset, path, attribute)
    _check_complete(attack, columns, path)
    return attack


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def format_interval(values: pd.Series, decimals: int = 2) -> str:
    values = values.dropna()
    if values.empty:
        raise ValueError("Cannot format an empty interval.")
    return f"$[{values.min():.{decimals}f}, {values.max():.{decimals}f}]$"


def format_max(values: pd.Series, decimals: int = 1) -> str:
    values = values.dropna()
    if values.empty:
        raise ValueError("Cannot format an empty maximum.")
    return f"${values.max():.{decimals}f}$"


def format_gamma_below(row: pd.Series, kind: str) -> str:
    """Models below the threshold after the attack, and how many the attack put
    there: the count in parentheses were at or above it before the injection."""
    below = int(row[f"gamma_below_{kind}"])
    induced = int(row[f"gamma_below_induced_{kind}"])
    # With nothing below the threshold there is nothing the attack put there,
    # so the parenthesis carries no information and is dropped.
    if below == 0:
        return "0"
    return f"{below} ({induced})"


def format_sig(row: pd.Series, kind: str) -> str:
    """Models reaching the threshold uncorrected, and how many survive the
    correction, in parentheses. Dropped where nothing reached it."""
    uncorrected = int(row[f"sig_uncorrected_{kind}"])
    corrected = int(row[f"sig_{kind}"])
    if uncorrected == 0:
        return "0"
    return f"{uncorrected} ({corrected})"


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------


def load_significance(
    attacks: list[tuple[str, str]],
    row_loader: RowLoader,
    budget: float,
    attribute: str,
) -> pd.DataFrame:
    rows = []
    for dataset in datasets_for(attribute):
        for attack_id, _ in attacks:
            attack = row_loader(dataset, attack_id, attribute, budget)
            for _, row in attack.iterrows():
                rows.append(
                    {
                        "dataset": dataset,
                        "attack_id": attack_id,
                        "model": row["model"],
                        "model_kind": model_kind(row["model"], attribute),
                        "p_value": float(row["p_value"]),
                    }
                )

    significance = pd.DataFrame(rows)
    significance["significant_uncorrected"] = significance["p_value"] < ALPHA
    significance["p_value_bh"] = benjamini_hochberg(significance["p_value"])
    significance["significant_bh"] = significance["p_value_bh"] < ALPHA
    return significance


def significance_counts(
    significance: pd.DataFrame, dataset: str, attack_id: str
) -> dict[str, dict[str, int]]:
    """Significant models per family, before and after the correction.

    The uncorrected count says how many tests reached the threshold at all; the
    corrected one how many of those survive. Reporting only the second cannot
    distinguish a test that never fires from one whose hits are all removed.
    """
    subset = significance[
        significance["dataset"].eq(dataset) & significance["attack_id"].eq(attack_id)
    ]
    by_kind = subset.groupby("model_kind")[
        ["significant_uncorrected", "significant_bh"]
    ].sum()
    return {
        kind: {
            "uncorrected": int(by_kind["significant_uncorrected"].get(kind, 0)),
            "bh": int(by_kind["significant_bh"].get(kind, 0)),
        }
        for kind in KINDS
    }


# --------------------------------------------------------------------------
# row construction
# --------------------------------------------------------------------------


def _kind_summary(frame: pd.DataFrame, gamma_col: str) -> dict[str, str]:
    """Summarize one model family within one table row.

    ``frame`` carries the post-attack gamma in ``gamma_col`` alongside the
    ``delta_gamma`` and ``delta_m_pct`` columns added by ``summarize_row``.
    """
    below_threshold = frame[gamma_col] < GAMMA_THRESHOLD
    return {
        "gamma": format_interval(frame[gamma_col]),
        "delta_gamma": format_interval(frame["delta_gamma"]),
        "delta_m_max": format_max(frame["delta_m_pct"]),
        "gamma_below": int(below_threshold.sum()),
        # Of those below the threshold, the ones the injection put there. A
        # model already below it beforehand contributes to gamma_below without
        # the attack having changed anything.
        "gamma_below_induced": int(
            (
                below_threshold
                & (frame[gamma_col] - frame["delta_gamma"] >= GAMMA_THRESHOLD)
            ).sum()
        ),
        "n_models": int(len(frame)),
        # retained so that a figure quoted in the text can be attributed
        "gamma_min_model": str(frame.loc[frame[gamma_col].idxmin(), "model"]),
        "delta_m_max_model": str(frame.loc[frame["delta_m_pct"].idxmax(), "model"]),
        "delta_gamma_min_model": str(frame.loc[frame["delta_gamma"].idxmin(), "model"]),
    }


def _assemble(base: dict, per_kind: dict[str, dict], sig: dict[str, dict]) -> dict:
    row = dict(base)
    for kind in KINDS:
        data = per_kind[kind]
        row[f"gamma_{kind}"] = data["gamma"]
        row[f"delta_gamma_{kind}"] = data["delta_gamma"]
        row[f"delta_m_max_{kind}"] = data["delta_m_max"]
        row[f"gamma_below_{kind}"] = data["gamma_below"]
        row[f"gamma_below_induced_{kind}"] = data["gamma_below_induced"]
        row[f"n_models_{kind}"] = data["n_models"]
        row[f"gamma_min_model_{kind}"] = data["gamma_min_model"]
        row[f"delta_gamma_min_model_{kind}"] = data["delta_gamma_min_model"]
        row[f"delta_m_max_model_{kind}"] = data["delta_m_max_model"]
        row[f"sig_{kind}"] = sig[kind]["bh"]
        row[f"sig_uncorrected_{kind}"] = sig[kind]["uncorrected"]
    return row


def summarize_row(
    dataset: str,
    attack_id: str,
    attack_label: str,
    budget: float,
    row_loader: RowLoader,
    significance: pd.DataFrame,
    attribute: str,
) -> dict:
    gamma_col = gamma_column(attribute)
    clean = load_clean(dataset, attribute)
    attack = row_loader(dataset, attack_id, attribute, budget)
    merged = attack.merge(
        clean[["model", gamma_col]],
        on="model",
        how="left",
        suffixes=("_post", "_clean"),
        validate="one_to_one",
    )
    merged["delta_gamma"] = merged[f"{gamma_col}_post"] - merged[f"{gamma_col}_clean"]
    merged["delta_m_pct"] = 100 * merged.apply(
        lambda row: group_benefit_gap(row, attribute, groups_config(attribute)), axis=1
    )
    merged["kind"] = merged["model"].map(lambda m: model_kind(m, attribute))

    per_kind = {
        kind: _kind_summary(merged[merged["kind"].eq(kind)], f"{gamma_col}_post")
        for kind in KINDS
    }
    base = {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "attack_id": attack_id,
        "attack": attack_label,
    }
    return _assemble(
        base, per_kind, significance_counts(significance, dataset, attack_id)
    )


def build_table(
    attacks: list[tuple[str, str]],
    row_loader: RowLoader,
    budget: float,
    attribute: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    significance = load_significance(attacks, row_loader, budget, attribute)
    rows = [
        summarize_row(
            dataset, attack_id, attack_label, budget, row_loader, significance, attribute
        )
        for dataset in datasets_for(attribute)
        for attack_id, attack_label in attacks
    ]
    return pd.DataFrame(rows), significance


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

COLUMN_COUNT = 10


def metric_header(leading_columns: list[str]) -> list[str]:
    """The two-level metric header shared by both tables.

    Both layouts put the same eight metric columns after their descriptive ones,
    so both come through here; they used to carry a copy each, which had already
    drifted by a whitespace character.
    """
    leading_count = len(leading_columns)
    conventional_start = leading_count + 1
    conventional_end = leading_count + 4
    fair_start = leading_count + 5
    fair_end = leading_count + 8
    empty = " " + " ".join(["&"] * leading_count)
    gamma_threshold = rf"$\gamma<{GAMMA_THRESHOLD:.1f}$"
    return [
        (
            f"{empty} \\multicolumn{{4}}{{c}}{{\\textbf{{Conventional}} (8 models)}}"
            r" & \multicolumn{4}{c}{\textbf{Fairness-aware} (5 models)} \\"
        ),
        (
            rf"\cmidrule(lr){{{conventional_start}-{conventional_end}}}"
            rf"\cmidrule(lr){{{fair_start}-{fair_end}}}"
        ),
        (
            " & ".join(rf"\textbf{{{column}}}" for column in leading_columns)
            + r" & $\Delta\gamma$ & $\Delta m^{\max}$ (\%) & "
            + gamma_threshold
            + r" & Sig. & $\Delta\gamma$ & $\Delta m^{\max}$ (\%) & "
            + gamma_threshold
            + r" & Sig. \\"
        ),
        r"\midrule",
    ]


def metric_cells(row: pd.Series) -> list[str]:
    """The four displayed cells for each model family, conventional first."""
    cells = []
    for kind in KINDS:
        cells.extend(
            [
                row[f"delta_gamma_{kind}"],
                row[f"delta_m_max_{kind}"],
                format_gamma_below(row, kind),
                format_sig(row, kind),
            ]
        )
    return cells


def _table_open() -> list[str]:
    return [
        r"\begin{table*}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\centering",
        r"\begin{tabular}{l l cccc cccc}",
        r"\toprule",
    ]


def _table_close(caption: str = "", label: str = "") -> list[str]:
    lines = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [
        r"\end{table*}",
        "",
    ]
    return lines


def write_latex_table(
    df: pd.DataFrame, path: Path, caption: str = "", label: str = ""
) -> None:
    """One row per attack, grouped by dataset. Used by the vanilla table."""
    rows_per_dataset = int(df.groupby("dataset").size().max())
    lines = _table_open()
    lines += metric_header(["Dataset", "Attack"])

    for dataset_idx, (_, group) in enumerate(df.groupby("dataset", sort=False)):
        if dataset_idx > 0:
            lines.append(rf"\cmidrule(lr){{1-{COLUMN_COUNT}}}")
        for row_idx, (_, row) in enumerate(group.iterrows()):
            prefix = (
                rf"\multirow{{{rows_per_dataset}}}{{*}}{{{row['dataset_label']}}} "
                if row_idx == 0
                else ""
            )
            lines.append(
                " & ".join([prefix + "& " + row["attack"], *metric_cells(row)]) + r" \\"
            )

    lines += _table_close(caption, label)
    path.write_text("\n".join(lines))


def write_cfair_table(
    df: pd.DataFrame, path: Path, caption: str = "", label: str = ""
) -> None:
    """All six attacks as one table, keyed by target and selected items."""
    lines = _table_open()
    lines.extend(metric_header(["Dataset", "Attack"]))

    rows_per_dataset = len(CFAIR_TARGET_GROUPS) + sum(
        len(selected) for _, selected in CFAIR_TARGET_GROUPS
    )
    for dataset_idx, (_, dataset_rows) in enumerate(df.groupby("dataset", sort=False)):
        if dataset_idx > 0:
            lines.append(rf"\cmidrule(lr){{1-{COLUMN_COUNT}}}")
        dataset_label = dataset_rows.iloc[0]["dataset_label"]
        by_attack = dataset_rows.set_index("attack_id")
        first_row_of_dataset = True
        for group_idx, (target, selected) in enumerate(CFAIR_TARGET_GROUPS):
            if group_idx == CFAIR_FIRST_NUKE_GROUP:
                lines.append(rf"\cmidrule(lr){{2-{COLUMN_COUNT}}}")
            elif group_idx > 0:
                lines.append(r"\addlinespace[2pt]")
            dataset_cell = (
                rf"\multirow{{{rows_per_dataset}}}{{*}}{{{dataset_label}}}"
                if first_row_of_dataset
                else ""
            )
            lines.append(
                f"{dataset_cell} & "
                rf"\multicolumn{{{COLUMN_COUNT - 1}}}{{l}}{{{target}}} \\"
            )
            first_row_of_dataset = False
            for attack_id, profile in selected:
                lines.append(
                    " & ".join(
                        [
                            "",
                            rf"\quad {profile}",
                            *metric_cells(by_attack.loc[attack_id]),
                        ]
                    )
                    + r" \\"
                )

    lines.extend(_table_close(caption, label))
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def crossing_summary(table: pd.DataFrame) -> str:
    """Split the below-threshold counts into induced and pre-existing crossings.

    ``gamma_below`` counts models below the four-fifths threshold after the
    injection, whichever side of it they started on. Separating the two says how
    many of them the attack is responsible for.
    """
    header = f"  {'dataset':<8} {'attack':<24} "
    for kind in KINDS:
        header += f"{KIND_LABELS[kind].lower():<20}"
    lines = [header.rstrip()]
    totals = {kind: [0, 0, 0] for kind in KINDS}
    for _, row in table.iterrows():
        cells = ""
        for kind in KINDS:
            below = int(row[f"gamma_below_{kind}"])
            induced = int(row[f"gamma_below_induced_{kind}"])
            n = int(row[f"n_models_{kind}"])
            totals[kind] = [
                totals[kind][0] + below,
                totals[kind][1] + induced,
                totals[kind][2] + n,
            ]
            cells += f"{f'{below}/{n} ({induced} induced)':<20}"
        lines.append(
            f"  {row['dataset_label']:<8} {plain_label(row['attack']):<24} "
            f"{cells}".rstrip()
        )
    summed = ""
    for kind in KINDS:
        below, induced, n = totals[kind]
        summed += f"{f'{below}/{n} ({induced} induced)':<20}"
    lines.append(f"  {'total':<8} {'':<24} {summed}".rstrip())
    return "\n".join(lines)


def pre_attack_summary(attribute: str) -> str:
    """Pre-attack state per dataset, for the summary file rather than the table."""
    gamma_col = gamma_column(attribute)
    lines = [
        f"  {'dataset':<8} {'conventional':<22} {'below':<7} "
        f"{'fairness-aware':<22} {'below':<7}"
    ]
    for dataset in datasets_for(attribute):
        clean = load_clean(dataset, attribute)
        clean["kind"] = clean["model"].map(lambda m: model_kind(m, attribute))
        cells = []
        for kind in KINDS:
            frame = clean[clean["kind"].eq(kind)]
            below = int((frame[gamma_col] < GAMMA_THRESHOLD).sum())
            cells += [
                f"[{frame[gamma_col].min():.2f}, {frame[gamma_col].max():.2f}]",
                f"{below}/{len(frame)}",
            ]
        lines.append(
            f"  {DATASET_LABELS[dataset]:<8} {cells[0]:<22} {cells[1]:<7} "
            f"{cells[2]:<22} {cells[3]:<7}"
        )
    return "\n".join(lines)


def group_assignment_audit(
    attacks: list[tuple[str, str]],
    row_loader: RowLoader,
    budget: float,
    attribute: str,
) -> str:
    """Report models whose recorded group split differs from their own outcomes.

    ``result_utils.infer_unprivileged_privileged`` derives the split from a
    model's own pre-attack group scores, taking the lower-scoring group as
    unprivileged. The experiment runners apply it to conventional models only:
    a fairness-aware model reuses the split of its conventional counterpart, via
    ``train_utils.get_base_model_for``, so that an attack targets the same users
    with and without the fairness intervention.

    The two disagree when a fairness-aware model's intervention has reversed the
    pre-attack disparity, in which case its ratio reads privileged over
    unprivileged benefit and exceeds 1. The comparison is recomputed on every
    run, since retuning a model or adding a dataset can change which models fall
    into it.
    """
    gamma_col = gamma_column(attribute)
    groups = groups_config(attribute)
    group_a, group_b = groups[attribute]
    first_attack = attacks[0][0]
    flagged = []
    for dataset in datasets_for(attribute):
        clean = load_clean(dataset, attribute).set_index("model")
        recorded = row_loader(dataset, first_attack, attribute, budget).set_index("model")
        for model, row in clean.iterrows():
            own_unprivileged, _ = infer_unprivileged_privileged(row, attribute, groups)
            used_unprivileged = recorded.loc[model, "unprivileged_group"]
            if own_unprivileged is None or own_unprivileged == used_unprivileged:
                continue
            flagged.append(
                f"    {DATASET_LABELS[dataset]:<7} {model:<20} "
                f"gamma = {row[gamma_col]:.3f}, unprivileged = {used_unprivileged} "
                f"(ndcg_{group_a}={row[f'ndcg_{group_a}']:.4f}, "
                f"ndcg_{group_b}={row[f'ndcg_{group_b}']:.4f})"
            )
    if not flagged:
        return "  Every model's recorded group split matches its own outcomes."
    return (
        "  The fairness intervention of these models has reversed the pre-attack\n"
        "  disparity. They keep their conventional counterpart's group split, so\n"
        "  their gamma reads privileged over unprivileged benefit:\n"
        + "\n".join(flagged)
    )


def significance_summary(significance: pd.DataFrame, family: str) -> str:
    n = len(significance)
    unc = int(significance["significant_uncorrected"].sum())
    bh = int(significance["significant_bh"].sum())
    by_kind_unc = significance.groupby("model_kind")["significant_uncorrected"].sum()
    by_kind_bh = significance.groupby("model_kind")["significant_bh"].sum()
    totals = significance.groupby("model_kind").size()
    lines = [
        f"[{family}] model-wise tests: {n}",
        f"  uncorrected p<{ALPHA}: {unc} ({100 * unc / n:.1f}%)"
        f"   [chance expectation under the null: {ALPHA * n:.1f}]",
        f"  BH-significant:       {bh} ({100 * bh / n:.1f}%)",
        f"  smallest BH-adjusted p: {significance['p_value_bh'].min():.3g}",
    ]
    for kind in KINDS:
        if kind in totals:
            lines.append(
                f"  {KIND_LABELS[kind]:<15} n={int(totals[kind]):<4} "
                f"uncorrected={int(by_kind_unc.get(kind, 0)):<3} "
                f"BH={int(by_kind_bh.get(kind, 0))}"
            )
    return "\n".join(lines)


ATTACK_FAMILY_CAPTIONS = {
    "vanilla": "Effect of vanilla shilling attacks",
    "cfair": "Effect of fairness shilling attacks",
}


def table_caption(
    significance: pd.DataFrame, family_key: str, attribute: str, budget: float
) -> str:
    """The caption, with the correction scope read off the tests it corrected.

    The number of model-wise tests is attribute-dependent (age covers fewer
    datasets than gender), so it is taken from ``significance`` rather than
    written by hand. Nothing else in the caption may state a count.
    """
    n_tests = len(significance)
    models_per_kind = significance.groupby("model_kind")["model"].nunique()
    kind_phrase = " and ".join(
        f"{int(models_per_kind[kind])} {KIND_LABELS[kind].lower()}"
        for kind in KINDS
        if kind in models_per_kind
    )
    lead = ATTACK_FAMILY_CAPTIONS[family_key]
    grouping = (
        r"Rows group each dataset's attacks by target-item strategy "
        r"(\textsc{PushR}, \textsc{PushLF}, \textsc{NukeF}). "
        if family_key == "cfair"
        else ""
    )
    return (
        f"{lead} at a {budget * 100:.0f}\\% budget on C-fairness when "
        f"{attribute} is used as the sensitive attribute, across {kind_phrase} "
        f"recommendation models. {grouping}"
        r"$\Delta\gamma$ reports the min--max attack-induced change in benefit "
        r"ratio within each model family, and $\Delta m^{\max}$ the largest "
        r"absolute post-attack group-benefit difference in percentage points. "
        rf"$\gamma < {GAMMA_THRESHOLD:.1f}$ counts models below the four-fifths "
        r"threshold after the attack; parentheses indicate those pushed below "
        r"it by the attack. Sig.\ counts models with a significant C-fairness "
        rf"reduction ($p < {ALPHA}$); parentheses indicate those remaining "
        r"significant after BH correction across all "
        rf"{n_tests} tests."
    )


def emit(
    table: pd.DataFrame,
    significance: pd.DataFrame,
    output_dir: Path,
    stem: str,
    family: str,
    attribute: str,
    family_key: str,
    budget: float,
    audit: str = "",
    latex_writer: Callable[..., None] = write_latex_table,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    pvalues_path = output_dir / f"{stem}_pvalues.csv"
    tex_path = output_dir / f"{stem}.tex"
    summary_path = output_dir / f"{stem}_summary.txt"

    table.to_csv(csv_path, index=False)
    significance.to_csv(pvalues_path, index=False)
    latex_writer(
        table,
        tex_path,
        caption=table_caption(significance, family_key, attribute, budget),
        label=f"tab:{stem}",
    )

    summary = significance_summary(significance, family)
    summary = (
        f"{summary}\n\nThreshold crossings "
        f"(gamma < {GAMMA_THRESHOLD:.1f} after injection; "
        f"induced = at or above it before):\n{crossing_summary(table)}"
    )
    summary = (
        f"{summary}\n\nPre-attack state (no injection):\n"
        f"{pre_attack_summary(attribute)}"
    )
    if audit:
        summary = f"{summary}\n\nGroup-assignment audit:\n{audit}"
    summary_path.write_text(summary + "\n")

    display_columns = [
        "dataset_label",
        "attack",
        "delta_gamma_base",
        "gamma_below_base",
        "sig_base",
        "delta_gamma_fair",
        "gamma_below_fair",
        "sig_fair",
    ]
    print(table[display_columns].to_string(index=False))
    print()
    print(summary)
    for path in (csv_path, pvalues_path, tex_path, summary_path):
        print(f"Wrote {path}")


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def table_stem(family: str, attribute: str, budget_tag: int) -> str:
    """Every artifact names its attribute, gender included.

    No attribute is privileged by the filenames: reading a stem tells you which
    one it holds without having to know which was the original.
    """
    return f"{family}_attacks_{attribute}_b{budget_tag}"


# family -> (attacks, row loader, LaTeX writer, name used in the summary)
FAMILY_SPECS = {
    "vanilla": (VANILLA_ATTACKS, load_vanilla_rows, write_latex_table, "vanilla"),
    "cfair": (CFAIR_ATTACKS, load_cfair_rows, write_cfair_table, "C-fair"),
}


def build_for(
    attribute: str, requested_family: str, budget: float, output_dir: Path | None
) -> None:
    """Build every requested table for one sensitive attribute."""
    available = families_for(attribute)
    datasets = datasets_for(attribute)
    if requested_family == "both":
        families = available
    elif requested_family not in available:
        raise SystemExit(
            f"No {requested_family} results exist for {attribute!r}: it was run "
            f"only for {available}. The {attribute} experiments cover "
            f"{', '.join(datasets)}."
        )
    else:
        families = [requested_family]

    budget_tag = int(round(budget * 100))
    for family in families:
        attacks, loader, writer, label = FAMILY_SPECS[family]
        stem = table_stem(family, attribute, budget_tag)
        target = output_dir or paths.tables_dir(family)
        print(
            f"\n=== {label} table, attribute={attribute}, "
            f"datasets={', '.join(datasets)} ==="
        )
        table, significance = build_table(attacks, loader, budget, attribute)
        emit(
            table,
            significance,
            target,
            stem,
            f"{label}/{attribute}",
            attribute,
            family,
            budget,
            group_assignment_audit(attacks, loader, budget, attribute),
            latex_writer=writer,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=["vanilla", "cfair", "both"],
        default="both",
        help="Which attack table to build.",
    )
    parser.add_argument(
        "--attribute",
        choices=sorted(ATTRIBUTE_CONFIGS) + ["all"],
        default=DEFAULT_ATTRIBUTE,
        help=(
            "Sensitive attribute. 'all' builds every attribute, each for the "
            "families it was actually run for."
        ),
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=0.1,
        help="Attack budget to include in the table.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated CSV and LaTeX table files.",
    )
    args = parser.parse_args()

    attributes = (
        sorted(ATTRIBUTE_CONFIGS) if args.attribute == "all" else [args.attribute]
    )
    if len(attributes) > 1 and args.output_dir is not None:
        raise SystemExit(
            "--output-dir with more than one attribute would collide; "
            "build one attribute at a time when overriding the output directory."
        )
    for attribute in attributes:
        family = args.family
        # Skip silently-impossible combinations rather than aborting the run,
        # but say so: a family missing for one attribute is not an error when
        # the user asked for all of them.
        if args.attribute == "all" and family != "both":
            if family not in families_for(attribute):
                print(f"skipping {attribute}/{family}: not run for this attribute")
                continue
        build_for(attribute, family, args.budget, args.output_dir)


if __name__ == "__main__":
    main()
