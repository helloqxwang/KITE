"""Lightweight embodiment support registry for experiment entrypoints."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBODIMENTS_CONFIG = EXPERIMENT_DIR / "conf" / "shared" / "embodiments.yaml"


from paper_config import runtime_config


def load_embodiment_registry(path: str | Path | None = None) -> DictConfig:
    """Load the experiment embodiment support matrix."""

    registry_path = DEFAULT_EMBODIMENTS_CONFIG if path is None else Path(path).expanduser()
    if not registry_path.is_absolute():
        registry_path = (EXPERIMENT_DIR / registry_path).resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"Embodiment registry not found: {registry_path}")
    registry = runtime_config(OmegaConf.load(registry_path))
    if OmegaConf.select(registry, "embodiments", default=None) is None:
        raise ValueError(f"Embodiment registry is missing an 'embodiments' section: {registry_path}")
    return registry


def embodiment_entry(key: str, registry: DictConfig | None = None) -> DictConfig:
    """Return one embodiment entry or raise with the available keys."""

    registry_cfg = load_embodiment_registry() if registry is None else registry
    key_text = str(key).strip()
    entry = OmegaConf.select(registry_cfg, f"embodiments.{key_text}", default=None)
    if entry is None:
        available = ", ".join(sorted(str(name) for name in registry_cfg.embodiments.keys()))
        raise ValueError(f"Unknown embodiment {key_text!r}. Available embodiments: {available}")
    return entry


def supported_capabilities(key: str, registry: DictConfig | None = None) -> tuple[str, ...]:
    """Return the support capability tags declared for an embodiment."""

    entry = embodiment_entry(key, registry=registry)
    return tuple(str(item) for item in OmegaConf.select(entry, "supported_by", default=[]))


def validate_embodiment_supported(
    key: str,
    capability: str,
    *,
    registry: DictConfig | None = None,
    label: str = "embodiment",
) -> DictConfig:
    """Validate that an embodiment is supported by a specific experiment path."""

    registry_cfg = load_embodiment_registry() if registry is None else registry
    key_text = str(key).strip()
    capability_text = str(capability).strip()
    from paper_config import CAPABILITY_NAMES
    capability_text = CAPABILITY_NAMES.get(capability_text, capability_text)
    entry = embodiment_entry(key_text, registry=registry_cfg)
    supported = supported_capabilities(key_text, registry=registry_cfg)
    if capability_text not in supported:
        supported_text = ", ".join(supported) if supported else "none"
        raise ValueError(
            f"{label}={key_text!r} does not support {capability_text!r}. "
            f"Supported capabilities: {supported_text}"
        )
    return entry


def display_name(key: str, registry: DictConfig | None = None) -> str:
    """Return a human-readable display name for README/status messages."""

    entry = embodiment_entry(key, registry=registry)
    return str(OmegaConf.select(entry, "display_name", default=key))
