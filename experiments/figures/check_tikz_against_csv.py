"""Audit a written TikZ figure against the result CSVs.

Deliberately independent of attack_figures.py: it re-reads the CSVs its own way
and compares every plotted point, so a figure pasted into the paper by hand --
or generated before a result was recomputed -- is caught rather than trusted.

    python experiments/figures/check_tikz_against_csv.py path/to/figure.tex \
        --attack-set favorite \
        --datasets ml-100k ml-1m lastfm-1k ftky fnyc \
        --models mf mfvalue mfabsolute mfunder mfover

Pass --datasets and --models explicitly: the defaults cover only ML100K, ML1M
and the conventional MF family, so a figure audited on the defaults alone leaves
rows and curves unchecked. --attack-set has to match the figure's columns, and
names the same sets as attack_figures.py.

The parser expects comments such as:

    % ===================== Row 1: ML100K =====================

and PGFPlots blocks such as:

    \\addplot[mf] table {
    0 0.99
    10 0.90
    };
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from src.helper_functions import paths
from src.helper_functions.experiment_registry import DATASET_LABELS, plain_label
from src.helper_functions.result_utils import canonical_run_keys
from experiments.figures.attack_figures import ATTACK_SETS, DEFAULT_ATTACK_SET


# The figures label rows by dataset; parsing one turns the label back into an id.
DATASET_IDS = {label: dataset for dataset, label in DATASET_LABELS.items()}


def attacks_by_column(attack_set: str) -> dict[int, tuple[str, str]]:
    """Which attack each column holds, left to right.

    The ids come from attack_figures so the two cannot disagree about what an
    attack is called; everything the checker actually compares -- the CSV
    lookup, the values, the tolerance -- stays its own.
    """
    return {
        index: (plain_label(title), attack_id)
        for index, (title, attack_id) in enumerate(ATTACK_SETS[attack_set]["attacks"])
    }


MODEL_ALIASES = {
    # Neighborhood models: the plot styles drop the hyphen the CSVs use. Without
    # these the checker reports MISSING_IN_CSV for every neighborhood figure.
    "slimu": "slim-u",
    "easeu": "ease-u",
    "slimi": "slim-i",
    "easei": "ease-i",
    "mfvalue": "mf-value",
    "mfabsolute": "mf-absolute",
    "mfunder": "mf-under",
    "mfover": "mf-over",
    "bnslimu": "bnslim-u",
}

BUDGET_TO_CSV = {
    0: 0.0,
    10: 0.1,
    25: 0.25,
    50: 0.5,
    75: 0.75,
    100: 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check written TikZ addplot values against result CSVs."
    )
    parser.add_argument("tex_path", type=Path)
    parser.add_argument(
        "--attack-set",
        default=DEFAULT_ATTACK_SET,
        choices=sorted(ATTACK_SETS),
        help="Which attack set the figure's three columns hold.",
    )
    parser.add_argument("--attribute", default="gender", choices=["gender", "age"])
    parser.add_argument("--metric", default="ndcg")
    parser.add_argument("--list-size", type=int, default=10)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ml-100k", "ml-1m"],
        help="Dataset ids to check, e.g. ml-100k ml-1m.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svd", "mf", "nmf", "neumf"],
        help="Plot model names to check.",
    )
    parser.add_argument("--atol", type=float, default=1e-12)
    return parser.parse_args()


def parse_tikz_tables(
    tex: str, columns: dict[int, tuple[str, str]]
) -> dict[tuple[str, str, str], dict[int, float]]:
    """Return {(dataset_id, attack_label, model): {budget_percent: value}}."""
    rows: dict[tuple[str, str, str], dict[int, float]] = {}
    current_dataset = None
    column_index = -1

    row_re = re.compile(r"Row\s+\d+:\s*([A-Za-z0-9-]+)")
    addplot_re = re.compile(
        r"\\addplot\[([^\]]+)\]\s+table\s*\{(?P<body>.*?)\};",
        re.DOTALL,
    )

    pos = 0
    while pos < len(tex):
        row_match = re.search(r"%\s*=+\s*Row\s+\d+:\s*([A-Za-z0-9-]+)\s*=+", tex[pos:])
        plot_match = re.search(r"\\nextgroupplot", tex[pos:])

        if row_match and (not plot_match or row_match.start() < plot_match.start()):
            label = row_re.search(row_match.group(0)).group(1)
            current_dataset = DATASET_IDS.get(label, label.lower())
            column_index = -1
            pos += row_match.end()
            continue

        if not plot_match:
            break

        plot_start = pos + plot_match.start()
        next_marker = re.search(
            r"%\s*=+\s*Row\s+\d+:|\\nextgroupplot|\\end\{groupplot\}",
            tex[plot_start + 1 :],
        )
        plot_end = len(tex) if not next_marker else plot_start + 1 + next_marker.start()
        block = tex[plot_start:plot_end]
        column_index += 1

        if current_dataset is not None and column_index in columns:
            attack_label, _attack_id = columns[column_index]
            for addplot in addplot_re.finditer(block):
                model = addplot.group(1).strip()
                values = {}
                for line in addplot.group("body").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    budget = int(float(parts[0]))
                    values[budget] = float(parts[1])
                rows[(current_dataset, attack_label, model)] = values

        pos = plot_end

    return rows


def csv_values(dataset: str, attack_id: str, model: str, attribute: str, metric: str, list_size: int):
    base = pd.read_csv(paths.clean_results(dataset, list_size))
    attack_path = paths.attack_results(dataset, attack_id, list_size)
    if not attack_path.exists():
        return None, f"missing CSV {attack_path}"
    attack = pd.read_csv(attack_path)
    attack = attack[attack["attribute"].eq(attribute)]

    # An index matching nothing leaves the rows alone: the checker reports on
    # what a figure plotted, so it must not empty the frame it is comparing.
    keys = canonical_run_keys(dataset, attack_id, attribute)
    if keys and "run_key" in attack.columns:
        attack = attack[attack["run_key"].isin(keys)]

    ratio_col = f"{metric}_{attribute}_ratio"
    values = {}
    for budget_percent, budget_csv in BUDGET_TO_CSV.items():
        if budget_percent == 0:
            rows = base[base["model"].eq(model)]
        else:
            rows = attack[attack["model"].eq(model) & attack["budget"].eq(budget_csv)]
            # dropna=False on purpose: one row holding the value and another
            # holding NaN is exactly the stale-row case, and taking the first
            # would be a coin flip.
            if len(rows) > 1 and rows[ratio_col].nunique(dropna=False) > 1:
                return None, (
                    f"ambiguous rows model={model} budget={budget_csv}: "
                    f"{sorted(rows[ratio_col].unique(), key=lambda v: (pd.isna(v), v))}"
                )
        if rows.empty:
            return None, f"missing row model={model} budget={budget_csv}"
        value = rows.iloc[0][ratio_col]
        if pd.isna(value):
            return None, f"NaN value model={model} budget={budget_csv}"
        values[budget_percent] = float(value)
    return values, None


def format_values(values: dict[int, float]) -> str:
    return " ".join(f"{budget}:{values[budget]:.15g}" for budget in sorted(values))


def csv_model_name(plot_model: str, attribute: str) -> str:
    base = MODEL_ALIASES.get(plot_model, plot_model)
    if base in {"mf-value", "mf-absolute", "mf-under", "mf-over", "bnslim-u"}:
        return f"{base}-{attribute}"
    return base


def main() -> None:
    args = parse_args()
    tex = args.tex_path.read_text()
    columns = attacks_by_column(args.attack_set)
    written = parse_tikz_tables(tex, columns)

    failures = 0
    for dataset in args.datasets:
        print(f"\n=== {dataset} / {args.attribute} / {args.attack_set} ===")
        for column, (attack_label, attack_id) in columns.items():
            print(f"\n{attack_label} ({attack_id})")
            for plot_model in args.models:
                model = csv_model_name(plot_model, args.attribute)
                key = (dataset, attack_label, plot_model)
                if key not in written:
                    failures += 1
                    print(f"  {plot_model}: MISSING_IN_TEX")
                    continue

                csv, error = csv_values(
                    dataset, attack_id, model, args.attribute, args.metric, args.list_size
                )
                if error:
                    failures += 1
                    print(f"  {plot_model} -> {model}: MISSING_IN_CSV ({error})")
                    print(f"    written: {format_values(written[key])}")
                    continue

                diffs = {
                    budget: written[key].get(budget, math.nan) - csv[budget]
                    for budget in BUDGET_TO_CSV
                }
                ok = all(abs(diff) <= args.atol for diff in diffs.values())
                print(f"  {plot_model} -> {model}: {'MATCH' if ok else 'DIFF'}")
                if not ok:
                    failures += 1
                    print(f"    written: {format_values(written[key])}")
                    print(f"    csv:     {format_values(csv)}")
                    bad = {
                        budget: diff
                        for budget, diff in diffs.items()
                        if abs(diff) > args.atol
                    }
                    print(
                        "    diffs:   "
                        + " ".join(f"{budget}:{diff:+.6g}" for budget, diff in bad.items())
                    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
