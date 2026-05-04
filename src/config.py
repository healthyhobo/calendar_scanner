"""Load and validate `config.yaml`.

This helper is tolerant of a few common issues that cause confusing
errors in notebooks and scripts:
 - If `pyyaml` is missing, it raises a clear ImportError.
 - If a relative path is provided, it will search upwards from the
   current working directory for the file (useful when the notebook's
   cwd is not the repo root).
 - If the file is missing or empty, it raises a helpful exception.
"""
from pathlib import Path
from typing import Union
import os

try:
    import yaml
except Exception:  # pragma: no cover - missing dependency
    yaml = None

# Optional: load environment variables from a .env file if python-dotenv is
# installed. This lets users keep sensitive tokens out of config.yaml.
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    # Provide a small fallback loader so `.env` files are still usable even
    # when `python-dotenv` is not installed (useful in minimal venvs).
    def load_dotenv(dotenv_path=None, override=False):
        try:
            if dotenv_path is None:
                return False
            p = Path(dotenv_path)
            if not p.exists():
                return False
            with p.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Remove surrounding quotes if present
                    if len(v) >= 2 and v[0] in ('"', "\'") and v[-1] == v[0]:
                        v = v[1:-1]
                    if override or (k not in os.environ):
                        os.environ[k] = v
            return True
        except Exception:
            return False


def load_config(path: Union[str, Path, None] = None) -> dict:
    if yaml is None:
        raise ImportError("Missing dependency: pyyaml. Install with: pip install pyyaml")

    # Default to repository-level config next to project root
    default_path = Path(__file__).resolve().parent.parent / "config.yaml"

    # If python-dotenv is available, try to load a `.env` file from the repo
    # root so environment variables (e.g., `ORATS_TOKEN`) can override
    # sensitive values in `config.yaml`.
    repo_root = default_path.parent
    if load_dotenv is not None:
        try:
            load_dotenv(dotenv_path=str(repo_root / ".env"))
        except Exception:
            # Do not fail if .env loading fails; env vars may still be set
            pass

    if path is None:
        cfg_path = default_path
    else:
        p = Path(path)
        if p.is_absolute() and p.exists():
            cfg_path = p
        else:
            # Search upward from current working directory for the given filename
            search_name = p.name
            cfg_path = None
            for d in [Path.cwd()] + list(Path.cwd().parents):
                candidate = d / search_name
                if candidate.exists():
                    cfg_path = candidate
                    break
            # Fallback to default location in the repo if not found
            if cfg_path is None:
                if default_path.exists():
                    cfg_path = default_path
                else:
                    raise FileNotFoundError(
                        f"Config file not found: looked for '{path}' from cwd and fallback {default_path}"
                    )

    # Read and validate YAML
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading config file {cfg_path}: {e}") from e

    if cfg is None:
        raise ValueError(f"Config file {cfg_path} is empty or contains invalid YAML")
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {cfg_path} did not load as a mapping (got {type(cfg)})")

    # Allow overriding sensitive tokens via environment variables.
    # If you set ORATS_TOKEN in your shell or in a .env file, it will replace
    # the value in config.yaml (if present). Ensure the returned config
    # always contains the `orats.token` key (may be None).
    orats_env_token = os.getenv("ORATS_TOKEN")
    orats_section = cfg.setdefault("orats", {})
    # Prefer environment variable if present, otherwise keep YAML value (or None)
    token_value = orats_env_token or orats_section.get("token")
    # Ensure key exists (explicitly set to None if missing)
    orats_section["token"] = token_value

    return cfg
