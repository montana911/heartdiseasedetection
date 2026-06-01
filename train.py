"""
Training entry point
====================

Trains the full pipeline from the two cleaned CSV files and saves the
resulting model to disk.

The trained pipeline will be written to ``models/pipeline.pkl`` and
the test-set evaluation will be printed to the console.
"""

import logging
import sys

from src import HeartDiseasePipeline, Config


def setup_logging() -> None:
    """Console logging — INFO level so the user sees stage progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    setup_logging()

    print("=" * 60)
    print("  Heart Disease Prediction — Training Pipeline")
    print("=" * 60)

    config   = Config()
    print(config.summary())

    pipeline = HeartDiseasePipeline(config=config)

    try:
        pipeline.fit_full_pipeline()
    except FileNotFoundError as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        return 1

    print("\n" + pipeline.summary())

    save_path = pipeline.save()
    print(f"\nTrained pipeline saved to: {save_path}")
    print("You can now run `streamlit run app.py` to launch the web app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
