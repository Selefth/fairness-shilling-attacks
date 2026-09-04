"""How model variants are named in result files, and how to read those names back.

A fairness-aware model is stored under its own name with the protected attribute
appended (``mf`` -> ``mf-value-gender``), and it borrows the privileged /
unprivileged split of a conventional counterpart so that an attack targets the
same users with and without the fairness intervention.

Those two conventions are needed both by the experiment runners, which write the
result files, and by the table scripts, which read them back. 
"""

BASE_MODEL_MAP = {
    "bnslim-u": "slim-u",
    "mf-value": "mf",
    "mf-absolute": "mf",
    "mf-under": "mf",
    "mf-over": "mf",
    # Add more mappings if needed
}


def get_base_model_for(model_name):
    """
    Returns the corresponding base model name for a fairness-aware model.
    If not found in mapping, returns the original model name.
    """
    return BASE_MODEL_MAP.get(model_name, model_name)


def model_result_name(model_kind, model_name, attribute):
    """Return the model label: base models use their name, fairness-aware models append the protected attribute."""
    return model_name if model_kind == "base" else f"{model_name}-{attribute}"


def model_kind_from_result_name(model_label, attribute):
    """Inverse of ``model_result_name``: classify a stored label as base or fair.

    Only labels carrying ``attribute`` are fairness-aware, so a run over one
    attribute does not pick up the models of another.
    """
    return "fair" if model_label.endswith(f"-{attribute}") else "base"


def strip_attribute_suffix(model_label, attribute):
    """Recover the model name a fairness-aware label was built from.

    ``model_result_name`` appends the attribute; this removes it, leaving a
    conventional label untouched.
    """
    suffix = f"-{attribute}"
    if model_label.endswith(suffix):
        return model_label[: -len(suffix)]
    return model_label
