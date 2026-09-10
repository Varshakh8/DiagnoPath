"""Prepare multi-disease Kaggle dataset for DiagnoPath AI Member 1."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_TRAIN = ROOT / "data" / "raw" / "training_data.csv"
RAW_TEST = ROOT / "data" / "raw" / "test_data.csv"
OUT_PATIENTS = ROOT / "data" / "patients.csv"
OUT_META = ROOT / "data" / "dataset_meta.json"
SOURCE_NOTE = ROOT / "data" / "SOURCE.txt"

# Features that typically need investigation / labs / imaging (higher cost)
INVESTIGATION_FEATURES = {
    "high_fever",
    "irregular_sugar_level",
    "yellowish_skin",
    "dark_urine",
    "yellow_urine",
    "yellowing_of_eyes",
    "acute_liver_failure",
    "fluid_overload",
    "swelling_of_stomach",
    "swelled_lymph_nodes",
    "blurred_and_distorted_vision",
    "bloody_stool",
    "obesity",
    "enlarged_thyroid",
    "brittle_nails",
    "knee_pain",
    "hip_joint_pain",
    "swelling_joints",
    "blood_in_sputum",
    "receiving_blood_transfusion",
    "coma",
    "stomach_bleeding",
    "distention_of_abdomen",
    "prominent_veins_on_calf",
    "palpitations",
    "silver_like_dusting",
    "inflammatory_nails",
    "family_history",
    "rusty_sputum",
    "mucoid_sputum",
    "abnormal_menstruation",
    "red_spots_over_body",
    "toxic_look_(typhos)",
    "altered_sensorium",
    "history_of_alcohol_consumption",
    "receiving_unsterile_injections",
}


def load_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Drop empty trailing columns from trailing commas
        while header and header[-1].strip() == "":
            header.pop()
        header = [h.strip() for h in header]
        label_col = header[-1]
        symptom_cols = header[:-1]

        rows: list[dict[str, str]] = []
        for raw in reader:
            if not raw:
                continue
            # pad/truncate to header length
            values = list(raw)
            while values and values[-1].strip() == "" and len(values) > len(header):
                values.pop()
            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            values = values[: len(header)]
            label = values[-1].strip()
            if not label:
                continue
            row = {symptom_cols[i]: values[i].strip() for i in range(len(symptom_cols))}
            row[label_col] = label
            rows.append(row)
    return header, rows


def main() -> None:
    header, train_rows = load_rows(RAW_TRAIN)
    _, test_rows = load_rows(RAW_TEST)
    label_col = header[-1]
    symptom_cols = header[:-1]

    # Combine train+test for richer demo; tag split
    all_rows: list[tuple[str, dict[str, str]]] = [
        ("train", r) for r in train_rows
    ] + [("test", r) for r in test_rows]

    OUT_PATIENTS.parent.mkdir(parents=True, exist_ok=True)
    out_fields = ["patient_id", "split"] + symptom_cols + ["disease"]

    with OUT_PATIENTS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, (split, row) in enumerate(all_rows, start=1):
            out = {
                "patient_id": f"P{i:04d}",
                "split": split,
                "disease": row[label_col],
            }
            for s in symptom_cols:
                out[s] = int(float(row[s] or 0))
            writer.writerow(out)

    counts = Counter(r[label_col] for _, r in all_rows)
    inquiry = [s for s in symptom_cols if s not in INVESTIGATION_FEATURES]
    investigations = [s for s in symptom_cols if s in INVESTIGATION_FEATURES]

    meta = {
        "source_name": "Disease Prediction Using Machine Learning (Kaggle)",
        "source_url": "https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning",
        "mirror_used": "https://github.com/anujdutt9/Disease-Prediction-from-Symptoms",
        "n_patients": len(all_rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_symptoms": len(symptom_cols),
        "n_diseases": len(counts),
        "diseases": sorted(counts.keys()),
        "disease_counts": dict(sorted(counts.items())),
        "inquiry_symptoms": inquiry,
        "investigation_features": investigations,
        "cost_policy": {
            "ask_symptom": 1,
            "order_investigation": 3,
        },
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    SOURCE_NOTE.write_text(
        "\n".join(
            [
                "Dataset source (Kaggle):",
                "https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning",
                "",
                "Downloaded via public GitHub mirror (same Training/Testing CSVs):",
                "https://github.com/anujdutt9/Disease-Prediction-from-Symptoms",
                "",
                f"Patients: {len(all_rows)} | Symptoms: {len(symptom_cols)} | Diseases: {len(counts)}",
                "",
                "Diseases:",
                *[f"- {d} ({counts[d]})" for d in sorted(counts)],
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUT_PATIENTS} ({len(all_rows)} patients)")
    print(f"Symptoms: {len(symptom_cols)} | Diseases: {len(counts)}")
    print(f"Inquiries: {len(inquiry)} | Investigations: {len(investigations)}")
    print("Top diseases:")
    for d, c in counts.most_common(8):
        print(f"  {d}: {c}")


if __name__ == "__main__":
    main()
