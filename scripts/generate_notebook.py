"""Generate a clean notebook from the reviewed source template, without analysis.

The template contains the current narrative and code with no saved execution.
The default output is a separate file, preserving the reviewed result notebook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notebooks/nlp_walkthrough.generated.ipynb")
    args = parser.parse_args()
    notebook = json.loads((ROOT / "notebooks/nlp_walkthrough.template.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
