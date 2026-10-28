"""Paired patient bootstrap of baseline versus Siamese predictions on the same patients.

    python -m recognition.Siamese_ISIC2020_Melanoma_47472442.compare \
        data/runs/baseline/final/predictions.csv data/runs/siamese/final/predictions.csv

Each draw resamples whole patients (stratified by whether they have a melanoma) and
scores both models on that draw, following Field and Welsh (2007),
https://doi.org/10.1111/j.1467-9868.2007.00593.x. Prints a Markdown table of
estimates with 95% percentile intervals.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_predictions(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        "image_name": np.array([r["image_name"] for r in rows]),
        "patient_id": np.array([r["patient_id"] for r in rows]),
        "label": np.array([int(r["actual"]) for r in rows]),
        "probability": np.array([float(r["probability"]) for r in rows]),
        "reference_score": np.array([float(r["reference_score"]) for r in rows]),
        "decision": np.array([r["decision"] for r in rows]),
    }


def summarise(run: dict[str, np.ndarray], index: np.ndarray) -> np.ndarray:
    """AUROC, AP, decision recall/precision/accuracy, review workload and melanoma capture."""
    labels, decision = run["label"][index], run["decision"][index]
    flagged, reviewed = decision == "melanoma", decision != "benign"
    return np.array([
        roc_auc_score(labels, run["probability"][index]),
        average_precision_score(labels, run["probability"][index]),
        flagged[labels == 1].mean(),
        labels[flagged].mean(),
        (flagged == labels).mean(),
        reviewed.mean(),
        reviewed[labels == 1].mean(),
    ])


NAMES = ("AUROC", "Melanoma AP", "Decision recall", "Decision precision", "Decision accuracy", "Review workload", "Review capture")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("siamese", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    baseline, siamese = read_predictions(args.baseline), read_predictions(args.siamese)

    images_by_patient: dict[str, list[int]] = defaultdict(list)
    for i, patient in enumerate(baseline["patient_id"]):
        images_by_patient[patient].append(i)
    has_melanoma = {p: baseline["label"][i].any() for p, i in images_by_patient.items()}
    strata = [np.array([p for p in images_by_patient if has_melanoma[p] == flag]) for flag in (True, False)]

    rng = np.random.default_rng(42)
    draws = []
    for _ in range(args.draws):
        patients = np.concatenate([rng.choice(group, len(group)) for group in strata])
        index = np.concatenate([images_by_patient[p] for p in patients])
        draws.append(np.stack([summarise(baseline, index), summarise(siamese, index)]))
    draws = np.array(draws)  # [draw, model, metric]
    everyone = np.arange(len(baseline["label"]))
    estimates = np.stack([summarise(baseline, everyone), summarise(siamese, everyone)])

    def interval(values: np.ndarray) -> str:
        low, high = np.nanpercentile(values, [2.5, 97.5])
        return f"[{low:.4f}, {high:.4f}]"

    print("| Measure | Baseline | Siamese | Siamese - baseline |")
    print("| --- | ---: | ---: | ---: |")
    for m, name in enumerate(NAMES):
        print(
            f"| {name} | {estimates[0, m]:.4f} {interval(draws[:, 0, m])} | {estimates[1, m]:.4f} {interval(draws[:, 1, m])} "
            f"| {estimates[1, m] - estimates[0, m]:+.4f} {interval(draws[:, 1, m] - draws[:, 0, m])} |"
        )


if __name__ == "__main__":
    main()
