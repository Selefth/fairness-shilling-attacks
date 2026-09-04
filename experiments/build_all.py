"""Rebuild the paper's tables and figures, exactly as the paper reports them.

    python experiments/build_all.py

Nothing here trains a model or runs an attack: it reads ``results/runs/`` and
writes ``results/derived/tables/`` and ``results/derived/figures/``.

What it builds:

    vanilla attack table          gender
    C-fair attack table           gender, age
    partial-knowledge table       gender
    budget-curve figures          4 model families x 2 attack sets x 2 attributes
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.helper_functions import paths
from src.helper_functions.experiment_registry import (
    ATTRIBUTE_CONFIGS,
    datasets_for,
    families_for,
)
from experiments.figures import attack_figures
from experiments.tables import attack_tables, partial_knowledge_table

# The budget every table in the paper reports. 
TABLE_BUDGET = 0.1


def plot_models(figure_name: str) -> list[str]:
    """The checker names models by their pgfplots style, which the figure owns."""
    return [style for style, _label, _base, _fair in attack_figures.FIGURES[figure_name]["models"]]


def build_tables() -> None:
    for attribute in sorted(ATTRIBUTE_CONFIGS):
        for family in families_for(attribute):
            attack_tables.build_for(attribute, family, TABLE_BUDGET, None)
    print("\n=== partial-knowledge table ===")
    partial_knowledge_table.build(budget=TABLE_BUDGET)


def build_figures() -> list[tuple[str, str, str]]:
    built = []
    for attribute in sorted(ATTRIBUTE_CONFIGS):
        for attack_set in sorted(attack_figures.ATTACK_SETS):
            for figure_name in sorted(attack_figures.FIGURES):
                tex, stats = attack_figures.generate(figure_name, attribute, attack_set)
                path = paths.figure_file(figure_name, attack_set, attribute)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(tex)
                attack_figures.report(path, stats, attribute)
                built.append((figure_name, attack_set, attribute))
    return built


def check_figures(built: list[tuple[str, str, str]]) -> int:
    """Audit each written figure against the CSVs, across every dataset it covers.

    The checker's own defaults cover two datasets and the conventional MF models,
    which would leave most of each figure unread -- so the full lists are passed
    here. A figure audited on the defaults alone is how a stale curve survived
    into the paper once already.
    """
    failures = 0
    for figure_name, attack_set, attribute in built:
        path = paths.figure_file(figure_name, attack_set, attribute)
        name = path.relative_to(paths.figures_dir()).with_suffix("")
        result = subprocess.run(
            [
                sys.executable,
                str(paths.REPO_ROOT / "experiments/figures/check_tikz_against_csv.py"),
                str(path),
                "--attack-set", attack_set,
                "--attribute", attribute,
                "--datasets", *datasets_for(attribute),
                "--models", *plot_models(figure_name),
            ],
            capture_output=True,
            text=True,
            cwd=paths.REPO_ROOT,
        )
        if result.returncode:
            failures += 1
            print(f"  FAILED  {name}")
            for line in result.stdout.splitlines():
                if "DIFF" in line or "MISSING" in line:
                    print(f"      {line.strip()}")
        else:
            print(f"  ok      {name}  ({result.stdout.count(': MATCH')} curves)")
    return failures


def main() -> None:
    # A parser with no arguments
    argparse.ArgumentParser(
        description="Rebuild the paper's tables and figures. Takes no arguments."
    ).parse_args()

    print(f"=== tables (budget {TABLE_BUDGET:.0%}) ===")
    build_tables()
    print("\n=== figures ===")
    built = build_figures()

    print("\n=== auditing every figure against the result CSVs ===")
    failures = check_figures(built)
    print(
        f"\n{len(built)} figures, {len(built) - failures} audited clean"
        + (f", {failures} FAILED" if failures else "")
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
