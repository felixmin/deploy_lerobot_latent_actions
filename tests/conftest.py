from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script_module(script_name: str):
    script_path = REPO_ROOT / "scripts" / script_name
    spec = spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
