"""
Command-line prediction entry point
===================================

Loads the trained pipeline and produces a single risk prediction for
a patient profile read from a JSON file or stdin. Useful for quick
sanity checks outside the web UI.

Usage
-----
    python predict.py path/to/patient.json
    cat patient.json | python predict.py
"""

from __future__ import annotations
import json
import sys
import logging
from pathlib import Path

from src import HeartDiseasePipeline


EXAMPLE_INPUT = {
    "age":               "60-64",
    "sex":               "Male",
    "race":              "White",
    "bmi":               28.5,
    "sleep_time":        6,
    "smoking":           "Yes",
    "alcohol":           "No",
    "stroke":            "No",
    "diff_walking":      "Yes",
    "physical_activity": "No",
    "asthma":            "No",
    "kidney_disease":    "No",
    "skin_cancer":       "No",
    "physical_health":   8,
    "mental_health":     2,
    "gen_health":        "Fair",
    "diabetic":          "Yes",
}


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    # Read patient JSON: file path arg, stdin pipe, or built-in example
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            patient = json.load(f)
    elif not sys.stdin.isatty():
        patient = json.load(sys.stdin)
    else:
        print("No input given. Using the built-in example profile.\n")
        patient = EXAMPLE_INPUT

    # Load the trained pipeline
    try:
        pipeline = HeartDiseasePipeline.load()
    except FileNotFoundError:
        print("No trained pipeline found. Run `python train.py` first.",
              file=sys.stderr)
        return 1

    probability, label = pipeline.predict_single(patient)

    print("Patient profile")
    print("-" * 40)
    for k, v in patient.items():
        print(f"  {k:<20}: {v}")

    print("\nPrediction")
    print("-" * 40)
    print(f"  Risk probability     : {probability:.1%}")
    print(f"  Risk band            : {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
