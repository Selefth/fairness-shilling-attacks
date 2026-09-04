"""Build the paper table of partial-knowledge C-fairness attacks.

The experiment runner writes one long-form row per dataset, model and attack
configuration to ``results/other_experiments/partial_knowledge/table_values.csv``.  This
script performs the paper-specific operations that do not belong in the runner:

* retain model--dataset pairs with at least one uncorrected p-value below 0.05;
* pivot Clean, Full, Sampling and the three Top-k variants into columns;
* mark raw significance levels and bold the lowest post-attack gamma; and
* write the wide values, p-value audit, summary and LaTeX table.

Run from anywhere with:

    python experiments/tables/partial_knowledge_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.helper_functions import paths
from src.helper_functions.experiment_registry import (
    ALPHA,
    DATASET_LABELS,
    DATASETS,
    DEFAULT_ATTRIBUTE,
)
from src.helper_functions.result_utils import benjamini_hochberg

ATTACK_COLUMNS = [
    ("none", "Clean", "clean"),
    ("cfair_influencer_push_random", "Full", "full"),
    ("cfair_sampling_push_random", "Sampling", "sampling"),
    ("cfair_influencer_top1_push_random", "Top-1", "top1"),
    ("cfair_influencer_top5_push_random", "Top-5", "top5"),
    ("cfair_influencer_top10_push_random", "Top-10", "top10"),
]
POST_ATTACKS = ATTACK_COLUMNS[1:]
ATTACK_IDS = [attack_id for attack_id, _, _ in ATTACK_COLUMNS]
POST_KEYS = [key for _, _, key in POST_ATTACKS]
ATTRIBUTE = DEFAULT_ATTRIBUTE
EXPECTED_MODELS_PER_DATASET = 13
EXPECTED_SAMPLING_SEEDS = "42;43;44"
REQUIRED_COLUMNS = {
    "dataset",
    "model",
    "model_kind",
    "attack_id",
    "attack",
    "budget",
    "runs",
    "seeds",
    "c_fairness",
    "p_value",
    "significant_p_0_05",
}


def _bool_series(values: pd.Series, source: Path) -> pd.Series:
    """Normalize a CSV boolean column without treating non-empty strings as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false"})
    if unknown:
        raise ValueError(
            f"Unexpected significant_p_0_05 values in {source}: {unknown}"
        )
    return normalized.eq("true")


def load_long_values(path: Path, budget: float) -> pd.DataFrame:
    """Load and validate the experiment rows used by the table."""
    values = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(values.columns))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    clean = values[
        values["attack_id"].eq("none")
        & np.isclose(values["budget"].astype(float), 0.0)
    ]
    post = values[
        values["attack_id"].isin(ATTACK_IDS[1:])
        & np.isclose(values["budget"].astype(float), budget)
    ]
    values = pd.concat([clean, post], ignore_index=True)

    unexpected_datasets = sorted(set(values["dataset"]) - set(DATASETS))
    if unexpected_datasets:
        raise ValueError(f"Unexpected datasets in {path}: {unexpected_datasets}")

    values["significant_p_0_05"] = _bool_series(
        values["significant_p_0_05"], path
    )
    if values["c_fairness"].isna().any():
        rows = values.loc[
            values["c_fairness"].isna(), ["dataset", "model", "attack_id"]
        ].to_dict("records")
        raise ValueError(f"Missing C-fairness values in {path}: {rows}")

    for dataset in DATASETS:
        dataset_rows = values[values["dataset"].eq(dataset)]
        clean_models = set(
            dataset_rows.loc[dataset_rows["attack_id"].eq("none"), "model"]
        )
        if len(clean_models) != EXPECTED_MODELS_PER_DATASET:
            raise ValueError(
                f"Expected {EXPECTED_MODELS_PER_DATASET} clean models for {dataset}, "
                f"found {len(clean_models)}: {sorted(clean_models)}"
            )
        for attack_id in ATTACK_IDS:
            attack_rows = dataset_rows[dataset_rows["attack_id"].eq(attack_id)]
            found_models = set(attack_rows["model"])
            duplicates = sorted(
                attack_rows.loc[
                    attack_rows.duplicated(["dataset", "model", "attack_id"], False),
                    "model",
                ].unique()
            )
            if found_models != clean_models or duplicates:
                raise ValueError(
                    f"Expected one {attack_id} row per clean model for {dataset}; "
                    f"missing={sorted(clean_models - found_models)}, "
                    f"unexpected={sorted(found_models - clean_models)}, "
                    f"duplicates={duplicates}"
                )

    attack_rows = values[values["attack_id"].ne("none")].copy()
    if attack_rows["p_value"].isna().any():
        rows = attack_rows.loc[
            attack_rows["p_value"].isna(), ["dataset", "model", "attack_id"]
        ].to_dict("records")
        raise ValueError(f"Missing post-attack p-values in {path}: {rows}")
    expected_flags = attack_rows["p_value"].lt(ALPHA)
    stale_flags = attack_rows["significant_p_0_05"].ne(expected_flags)
    if stale_flags.any():
        rows = attack_rows.loc[
            stale_flags,
            ["dataset", "model", "attack_id", "p_value", "significant_p_0_05"],
        ].to_dict("records")
        raise ValueError(f"Stored significance flags disagree with p-values: {rows}")

    sampling = attack_rows[
        attack_rows["attack_id"].eq("cfair_sampling_push_random")
    ]
    invalid_sampling = sampling[
        sampling["runs"].astype(int).ne(3)
        | sampling["seeds"].astype(str).ne(EXPECTED_SAMPLING_SEEDS)
    ]
    if not invalid_sampling.empty:
        rows = invalid_sampling[
            ["dataset", "model", "runs", "seeds"]
        ].to_dict("records")
        raise ValueError(f"Unexpected Sampling repetitions: {rows}")

    single_seed = attack_rows[
        attack_rows["attack_id"].ne("cfair_sampling_push_random")
    ]
    invalid_single_seed = single_seed[
        single_seed["runs"].astype(int).ne(1)
        | single_seed["seeds"].astype(str).ne("42")
    ]
    if not invalid_single_seed.empty:
        rows = invalid_single_seed[
            ["dataset", "model", "attack_id", "runs", "seeds"]
        ].to_dict("records")
        raise ValueError(f"Unexpected single-seed attack rows: {rows}")

    return values


def build_pvalue_audit(values: pd.DataFrame) -> pd.DataFrame:
    """Return all 325 raw tests plus a global BH audit and table inclusion."""
    audit = values[values["attack_id"].ne("none")].copy()
    audit["p_value_bh_all"] = benjamini_hochberg(audit["p_value"]).to_numpy()
    audit["significant_bh_all"] = audit["p_value_bh_all"].lt(ALPHA)
    included_pairs = set(
        audit.loc[audit["p_value"].lt(ALPHA), ["dataset", "model"]]
        .itertuples(index=False, name=None)
    )
    audit["included_in_table"] = [
        (dataset, model) in included_pairs
        for dataset, model in zip(audit["dataset"], audit["model"])
    ]
    columns = [
        "dataset",
        "model",
        "model_kind",
        "attack_id",
        "attack",
        "c_fairness",
        "p_value",
        "significant_p_0_05",
        "p_value_bh_all",
        "significant_bh_all",
        "included_in_table",
    ]
    audit = audit[columns].copy()
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    attack_order = {
        attack_id: index for index, (attack_id, _, _) in enumerate(POST_ATTACKS)
    }
    audit["_dataset_order"] = audit["dataset"].map(dataset_order)
    audit["_attack_order"] = audit["attack_id"].map(attack_order)
    audit = audit.sort_values(
        ["_dataset_order", "model", "_attack_order"], kind="stable"
    )
    return audit.drop(columns=["_dataset_order", "_attack_order"]).reset_index(
        drop=True
    )


def build_wide_table(values: pd.DataFrame) -> pd.DataFrame:
    """Select significant pairs and pivot the six configurations into columns."""
    lookup = values.set_index(["dataset", "model", "attack_id"])
    post = values[values["attack_id"].ne("none")]
    included = (
        post.groupby(["dataset", "model"], sort=False)["p_value"]
        .min()
        .lt(ALPHA)
    )
    included_pairs = included[included].index.tolist()

    rows = []
    for dataset, model in included_pairs:
        row = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "model": model,
        }
        for attack_id, _, key in ATTACK_COLUMNS:
            source = lookup.loc[(dataset, model, attack_id)]
            row[f"{key}_gamma"] = float(source["c_fairness"])
            if key != "clean":
                row[f"{key}_p_value"] = float(source["p_value"])
                row[f"{key}_significant"] = bool(source["p_value"] < ALPHA)
        row["lowest_post_attack"] = min(
            POST_KEYS, key=lambda key: row[f"{key}_gamma"]
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    dataset_order = {dataset: index for index, dataset in enumerate(DATASETS)}
    table["_dataset_order"] = table["dataset"].map(dataset_order)
    table = table.sort_values(["_dataset_order", "model"], kind="stable")
    return table.drop(columns="_dataset_order").reset_index(drop=True)


def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < ALPHA:
        return "*"
    return ""


def latex_cell(row: pd.Series, key: str) -> str:
    value = f"{row[f'{key}_gamma']:.3f}"
    if key == row["lowest_post_attack"]:
        value = rf"\textbf{{{value}}}"
    if key != "clean":
        stars = significance_stars(float(row[f"{key}_p_value"]))
        if stars:
            value += rf"$^{{{stars}}}$"
    return value


def write_latex(table: pd.DataFrame, path: Path, budget: float) -> None:
    budget_pct = int(round(100 * budget))
    lines = [
        r"\begin{table*}",
        r"\centering",
        (
            rf"\caption[C-fairness effects of partial-knowledge attacks at a "
            rf"${budget_pct}\%$ budget]{{Partial-knowledge C-fairness attacks at a "
            rf"${budget_pct}\%$ budget."
        ),
        (
            r"Results are reported only for model--dataset pairs where at least "
            r"one attack configuration causes a statistically significant "
            r"fairness degradation."
        ),
        (
            r"Values correspond to the benefit ratio $\gamma$ (C-fairness); "
            r"lower values indicate greater unfairness."
        ),
        (
            r"\textit{Clean} reports the C-fairness value at $0\%$ attack "
            r"budget, while \textit{Full} denotes the original full-knowledge "
            r"\textsc{PushR} C-Fair Influencer Attack at the stated budget."
        ),
        r"The lowest post-attack $\gamma$ in each row is shown in bold.",
        (
            r"Statistical significance is measured relative to the clean model "
            r"using uncorrected p-values ($^{*}p<0.05$, $^{**}p<0.01$, "
            r"$^{***}p<0.001$).}"
        ),
        r"\label{tab:partial_knowledge_attacks}",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Model & Clean & Full & Sampling & Top-1 & Top-5 & Top-10 \\",
        r"\midrule",
    ]

    for dataset_index, (_, group) in enumerate(table.groupby("dataset", sort=False)):
        if dataset_index > 0:
            lines.append(r"\cmidrule(lr){1-8}")
        for row_index, (_, row) in enumerate(group.iterrows()):
            dataset_cell = (
                rf"\multirow{{{len(group)}}}{{*}}{{{row['dataset_label']}}}"
                if row_index == 0
                else ""
            )
            cells = [latex_cell(row, key) for _, _, key in ATTACK_COLUMNS]
            lines.append(
                " & ".join([dataset_cell, str(row["model"]), *cells]) + r" \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def summary_text(
    source: Path, table: pd.DataFrame, pvalues: pd.DataFrame, budget: float
) -> str:
    try:
        source_display = source.resolve().relative_to(paths.REPO_ROOT)
    except ValueError:
        source_display = source
    selected_by_dataset = table.groupby("dataset", sort=False).size()
    lines = [
        "[partial knowledge]",
        f"  source: {source_display}",
        f"  budget: {budget:.3g}",
        f"  model-wise tests: {len(pvalues)}",
        f"  uncorrected p<{ALPHA}: {int(pvalues['significant_p_0_05'].sum())}",
        f"  BH-significant across all tests: {int(pvalues['significant_bh_all'].sum())}",
        f"  selected model--dataset pairs: {len(table)}",
        "  selected rows by dataset:",
    ]
    for dataset in DATASETS:
        lines.append(
            f"    {DATASET_LABELS[dataset]}: {int(selected_by_dataset.get(dataset, 0))}"
        )
    lines.extend(
        [
            "",
            "The LaTeX table uses uncorrected p-values for row selection and stars.",
            "The BH values are retained in the p-value audit but do not alter the table.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=paths.partial_knowledge_table_values(),
        help="Long-form values written by partial_knowledge_attacks.py.",
    )
    parser.add_argument(
        "--budget", type=float, default=0.1, help="Post-attack budget to include."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.tables_dir("partial_knowledge"),
        help="Directory for generated table artifacts.",
    )
    return parser.parse_args()


def build(input_path: Path | None = None, budget: float = 0.1,
          output_dir: Path | None = None) -> None:
    """Write every artifact of this table. Callable, so build_all can reuse it."""
    input_path = input_path or paths.partial_knowledge_table_values()
    output_dir = output_dir or paths.tables_dir("partial_knowledge")

    values = load_long_values(input_path, budget)
    table = build_wide_table(values)
    pvalues = build_pvalue_audit(values)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"partial_knowledge_attacks_{ATTRIBUTE}_b{int(round(100 * budget))}"
    csv_path = output_dir / f"{stem}.csv"
    tex_path = output_dir / f"{stem}.tex"
    pvalues_path = output_dir / f"{stem}_pvalues.csv"
    summary_path = output_dir / f"{stem}_summary.txt"

    table.to_csv(csv_path, index=False)
    pvalues.to_csv(pvalues_path, index=False)
    write_latex(table, tex_path, budget)
    summary_path.write_text(summary_text(input_path, table, pvalues, budget))

    print(table[["dataset_label", "model", "lowest_post_attack"]].to_string(index=False))
    for path in (csv_path, pvalues_path, tex_path, summary_path):
        print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    build(args.input, args.budget, args.output_dir)


if __name__ == "__main__":
    main()
