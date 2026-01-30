import argparse
import json
import itertools
from pathlib import Path

from config import RunConfig
import run_mna


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_list(value):
    if isinstance(value, list):
        return value
    return [value]


def _grid_from_dict(data):
    keys = list(data.keys())
    values = [_ensure_list(data[k]) for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _apply_config(config: RunConfig, overrides: dict, ignore_unknown: bool):
    for key, value in overrides.items():
        if not hasattr(config, key):
            if ignore_unknown:
                continue
            raise ValueError(f"Unknown config field: {key}")
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default="./condition/test.json")
    parser.add_argument("--base_output", default="./outputs/test")
    parser.add_argument("--ignore_unknown", action="store_true")
    args = parser.parse_args()

    base_output = Path(args.base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    data = _load_json(args.test_path)
    runs = list(_grid_from_dict(data))

    for idx, overrides in enumerate(runs, start=1):
        config = RunConfig()
        _apply_config(config, overrides, args.ignore_unknown)

        run_dir = base_output / str(idx)
        run_dir.mkdir(parents=True, exist_ok=True)
        config.output_path = run_dir

        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=4)

        run_mna.main_impl(config)


if __name__ == "__main__":
    main()
