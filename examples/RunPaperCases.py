"""
Run any of the 22 PR/SRK examples through discovery and TPD screening.
Read inputs from the case scripts; no previous results are required.
Save spinodals, corrected critical points and local/TPD assessments.
Report numerical failures and return a nonzero exit status.
"""

import argparse
from dataclasses import asdict
from importlib import import_module
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from HelmholtzCPCubic import AdaptiveBoundaryBellFinder, CriticalPointSolver
from HelmholtzCPCubic.Application import TPDStabilityClassifier
from examples import CASE_NAMES


def _jsonValue(value):
    """
    Convert result arrays while retaining nonfinite diagnostics explicitly.
    """

    if isinstance(value, np.ndarray):
        return _jsonValue(value.tolist())
    if isinstance(value, np.generic):
        return _jsonValue(value.item())
    if isinstance(value, dict):
        return {key: _jsonValue(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonValue(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> int:
    """
    Calculate named examples and save one JSON per case in K/bar/mol/L.

    With no case names, run C1C2_PR. Use --all for the complete example set,
    --list to show available names, and --output for a separate result folder.
    Existing result files are preserved. Return 1 if any search is incomplete
    or a candidate cannot be classified, and 0 otherwise.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", metavar="CASE")
    parser.add_argument("--all", action="store_true", dest="allCases")
    parser.add_argument("--list", action="store_true", dest="listCases")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "output" / "PaperCases")
    args = parser.parse_args()
    if args.listCases:
        print("\n".join(CASE_NAMES))
        return 0
    if args.allCases and args.cases:
        parser.error("choose named cases or --all")
    names = CASE_NAMES if args.allCases else tuple(args.cases) or ("C1C2_PR",)
    if any(name not in CASE_NAMES for name in names):
        parser.error("unknown case; use --list to see available names")
    if len(set(names)) != len(names):
        parser.error("case names must not be repeated")
    for name in names:
        if (args.output / f"{name}.json").exists():
            parser.error(f"preserve existing {name}.json; choose another --output folder")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    for name in names:
        destination = args.output / f"{name}.json"
        print(f"START {name}", flush=True)
        record = {"case": name}
        began = perf_counter()
        try:
            case = import_module(f"examples.{name}")
            model, composition, config = case.makeCase()
            solver = CriticalPointSolver(model, config)
            finder = AdaptiveBoundaryBellFinder(solver, case.traceConfig, stepConfig=case.stepConfig)
            discovery = finder.findBoundaryContours(composition)
            record["discoverySeconds"] = perf_counter() - began
            record["discovery"] = asdict(discovery)
            print(f"DISCOVERY {name}: completed={discovery.completed}, roots={len(discovery.roots)}", flush=True)
            destination.write_text(json.dumps(_jsonValue(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            classifier = TPDStabilityClassifier(model)
            record["screening"] = []
            began = perf_counter()
            for index, point in enumerate(discovery.roots):
                assessment = classifier.evaluate(point)
                record["screening"].append({
                    "candidate": point.asDict(), "status": assessment.status.value,
                    "lowestTpd": assessment.lowestTpdValue, "failure": assessment.failureReason,
                    "local": None if assessment.local is None else asdict(assessment.local),
                    "tpd": None if assessment.tpd is None else assessment.tpd.asDict(),
                })
                print(f"SCREEN {name} root {index + 1}: {assessment.status.value}, "
                      f"TPD={assessment.lowestTpdValue}, failure={assessment.failureReason}", flush=True)
            record["screeningSeconds"] = perf_counter() - began
            record["completed"] = discovery.completed and all(
                item["status"] != "failed" for item in record["screening"])
        except (ValueError, RuntimeError, FloatingPointError) as error:
            record.update(completed=False, error=f"{type(error).__name__}: {error}")
            print(f"FAILED {name}: {record['error']}", flush=True)
        destination.write_text(json.dumps(_jsonValue(record), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        completed.append(record["completed"])
        print(f"END {name}: completed={record['completed']}; {destination}", flush=True)
    return 0 if all(completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
