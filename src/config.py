"""Configuration loader for the customer churn baseline model."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REQUIRED_CONFIG_KEYS = ["seed", "paths", "split", "target", "id_col"]


def load_config(path: str = "configs/baseline.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If required configuration keys are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        msg = f"Configuration file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        msg = f"Invalid configuration format in {path}; expected dictionary."
        logger.error(msg)
        raise KeyError(msg)

    for key in REQUIRED_CONFIG_KEYS:
        if key not in cfg:
            msg = f"Missing required configuration key: '{key}' in {path}"
            logger.error(msg)
            raise KeyError(msg)

    return cfg
