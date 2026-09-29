"""Additional Laya-MLX checkpoints (e.g. fine-tuned ones) served next to the built-in models."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

ENV_VAR = "SYSTEMONE_MLX_MODELS"
LEGACY_ENV_VAR = "JEV_LOCAL_MLX_MODELS"  # former name, still read
RESERVED = {"jev-latest"}


def read_env(name: str, legacy: str) -> str:
    """Read a setting from its current name or the former JEV_LOCAL_* name; setting both differently is an error."""
    current, old = os.environ.get(name), os.environ.get(legacy)
    if current and old and current != old:
        raise ValueError(f"{name} and the former name {legacy} are both set with different values; unset {legacy}")
    return current or old or ""


def parse_spec(spec: str, taken: set[str]) -> dict[str, str]:
    """Parse `name=checkpoint,name2=checkpoint2` into {name: checkpoint}.

    A checkpoint is a local directory (converted with `laya-mlx convert`) or a Hugging Face id. Names must be
    unique and must not shadow a built-in model or `jev-latest`, because the request's `model` field picks
    the backend and there is no fallback between models.
    """
    models: dict[str, str] = {}
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, checkpoint = entry.partition("=")
        name, checkpoint = name.strip(), checkpoint.strip()
        if not sep or not name or not checkpoint:
            raise ValueError(f"{ENV_VAR} entries must look like name=checkpoint, got {entry!r}")
        if name in taken or name in RESERVED or name in models:
            raise ValueError(f"{ENV_VAR}: model name {name!r} is already in use")
        models[name] = checkpoint
    return models


def load_extra_backends(taken: set[str], factory: Callable[..., Any],
                        spec: str | None = None) -> dict[str, Any]:
    """Load every configured checkpoint; a checkpoint that cannot load fails startup instead of being skipped."""
    spec = read_env(ENV_VAR, LEGACY_ENV_VAR) if spec is None else spec
    return {name: factory(checkpoint, model_name=name, extend_context=False)
            for name, checkpoint in parse_spec(spec, taken).items()}


PROXY_ENV_VAR = "SYSTEMONE_PROXY_MODELS"
LEGACY_PROXY_ENV_VAR = "JEV_LOCAL_PROXY_MODELS"  # former name, still read


def parse_proxy_spec(spec: str, taken: set[str]) -> dict[str, tuple[str, str]]:
    """Parse `name=url[|upstream-model],...` into {name: (url, upstream_model)}.

    The upstream model defaults to `<first part of name>-latest` (kev-4b -> kev-latest, jeff-2b -> jeff-latest).
    Names follow the same rules as SYSTEMONE_MLX_MODELS. Use the names in the recommendation catalog
    (kev-4b, kev-0.8b, jeff-2b, jeff-0.8b) so /v1/recommend recognises them as available.
    """
    proxies: dict[str, tuple[str, str]] = {}
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        name, sep, target = entry.partition("=")
        name = name.strip()
        url, _, upstream = target.strip().partition("|")
        url, upstream = url.strip(), upstream.strip()
        if not sep or not name or not url.startswith(("http://", "https://")):
            raise ValueError(f"{PROXY_ENV_VAR} entries must look like name=http://host:port[|model], got {entry!r}")
        if name in taken or name in RESERVED or name in proxies:
            raise ValueError(f"{PROXY_ENV_VAR}: model name {name!r} is already in use")
        proxies[name] = (url, upstream or f"{name.split('-')[0]}-latest")
    return proxies


def load_proxy_backends(taken: set[str], factory: Callable[..., Any], spec: str | None = None,
                        warn: Callable[[str], None] = print) -> dict[str, Any]:
    """Register proxies. An unreachable upstream only warns at startup: it may be started later, and requests
    to it fail explicitly (502/503) until it is."""
    spec = read_env(PROXY_ENV_VAR, LEGACY_PROXY_ENV_VAR) if spec is None else spec
    backends = {}
    for name, (url, upstream) in parse_proxy_spec(spec, taken).items():
        backends[name] = factory(name, url, upstream)
        problem = backends[name].probe() if hasattr(backends[name], "probe") else None
        if problem:
            warn(f"warning: proxy model {name!r} upstream {url} is not answering yet ({problem})")
    return backends
