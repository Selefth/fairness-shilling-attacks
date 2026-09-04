import yaml


def load_attack_configs(path, config_group, class_registry):
    """Load attack configs and resolve attack class names to class objects."""
    with open(path, "r") as f:
        all_configs = yaml.safe_load(f)

    if config_group not in all_configs:
        raise ValueError(
            f"Unknown attack config group '{config_group}'. "
            f"Available groups: {sorted(all_configs)}"
        )

    configs = []
    for config in all_configs[config_group]:
        config = dict(config)
        class_name = config.pop("attack_class")
        if class_name not in class_registry:
            raise ValueError(
                f"Attack class '{class_name}' is not registered. "
                f"Available classes: {sorted(class_registry)}"
            )
        config["attack_cls"] = class_registry[class_name]
        config.setdefault("attack_config", {})
        configs.append(config)

    return configs
