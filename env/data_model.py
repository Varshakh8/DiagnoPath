"""
DiagnoPath AI - Member 1: Clinical Data Model
Loads the multi-disease Kaggle symptom dataset (stdlib only).
No built-in ML libraries used.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def default_data_path() -> Path:
    return _data_dir() / "patients.csv"


def default_meta_path() -> Path:
    return _data_dir() / "dataset_meta.json"


@dataclass
class DatasetMeta:
    """Vocabulary and cost policy loaded from dataset_meta.json."""

    source_name: str
    source_url: str
    n_patients: int
    n_symptoms: int
    n_diseases: int
    diseases: List[str]
    inquiry_symptoms: List[str]
    investigation_features: List[str]
    ask_cost: int = 1
    investigation_cost: int = 3

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DatasetMeta":
        path = path or default_meta_path()
        raw = json.loads(path.read_text(encoding="utf-8"))
        policy = raw.get("cost_policy", {})
        return cls(
            source_name=raw["source_name"],
            source_url=raw["source_url"],
            n_patients=raw["n_patients"],
            n_symptoms=raw["n_symptoms"],
            n_diseases=raw["n_diseases"],
            diseases=list(raw["diseases"]),
            inquiry_symptoms=list(raw["inquiry_symptoms"]),
            investigation_features=list(raw["investigation_features"]),
            ask_cost=int(policy.get("ask_symptom", 1)),
            investigation_cost=int(policy.get("order_investigation", 3)),
        )

    @property
    def all_features(self) -> List[str]:
        # Keep stable order: inquiries first, then investigations
        seen = set()
        ordered: List[str] = []
        for name in self.inquiry_symptoms + self.investigation_features:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered


# Module-level vocabulary (filled when meta/dataset loads)
META: DatasetMeta = DatasetMeta.load()
SYMPTOMS: List[str] = list(META.inquiry_symptoms)
MEDICAL_TESTS: Dict[str, Dict[str, object]] = {
    name: {
        "cost": META.investigation_cost,
        "label": name.replace("_", " ").title(),
    }
    for name in META.investigation_features
}
DISEASES: List[str] = list(META.diseases)
TEST_COSTS: Dict[str, int] = {
    name: int(meta["cost"]) for name, meta in MEDICAL_TESTS.items()
}
ASK_COST: int = META.ask_cost


@dataclass
class Patient:
    """One patient case used inside the diagnostic environment."""

    patient_id: str
    split: str
    symptoms: Dict[str, int]
    test_values: Dict[str, int]
    disease: str

    def true_label(self) -> str:
        return self.disease

    def present_symptoms(self) -> List[str]:
        return [k for k, v in self.symptoms.items() if v == 1]

    def present_investigations(self) -> List[str]:
        return [k for k, v in self.test_values.items() if v == 1]


@dataclass
class ClinicalDataset:
    """Loads and indexes multi-disease patient cases from CSV (stdlib only)."""

    patients: List[Patient] = field(default_factory=list)
    meta: DatasetMeta = field(default_factory=DatasetMeta.load)

    @classmethod
    def from_csv(
        cls,
        path: str | Path | None = None,
        meta: DatasetMeta | None = None,
    ) -> "ClinicalDataset":
        path = Path(path) if path else default_data_path()
        meta = meta or DatasetMeta.load()
        inquiry = meta.inquiry_symptoms
        investigations = meta.investigation_features
        patients: List[Patient] = []

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symptoms = {s: int(float(row[s])) for s in inquiry}
                test_values = {t: int(float(row[t])) for t in investigations}
                patients.append(
                    Patient(
                        patient_id=row["patient_id"],
                        split=row.get("split", "train"),
                        symptoms=symptoms,
                        test_values=test_values,
                        disease=row["disease"].strip(),
                    )
                )
        return cls(patients=patients, meta=meta)

    def get(self, index: int) -> Patient:
        return self.patients[index]

    def find_by_disease(self, disease: str) -> List[int]:
        return [i for i, p in enumerate(self.patients) if p.disease == disease]

    def first_index_for(self, disease: str) -> int:
        matches = self.find_by_disease(disease)
        if not matches:
            raise ValueError(f"No patient found for disease: {disease}")
        return matches[0]

    def __len__(self) -> int:
        return len(self.patients)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for p in self.patients:
            counts[p.disease] = counts.get(p.disease, 0) + 1
        return counts


def feature_cost(feature_name: str, meta: Optional[DatasetMeta] = None) -> int:
    meta = meta or META
    if feature_name in meta.investigation_features:
        return meta.investigation_cost
    return meta.ask_cost


def reload_vocabulary(meta: Optional[DatasetMeta] = None) -> DatasetMeta:
    """Refresh module-level lists after regenerating dataset_meta.json."""
    global META, SYMPTOMS, MEDICAL_TESTS, DISEASES, TEST_COSTS, ASK_COST
    META = meta or DatasetMeta.load()
    SYMPTOMS = list(META.inquiry_symptoms)
    MEDICAL_TESTS = {
        name: {
            "cost": META.investigation_cost,
            "label": name.replace("_", " ").title(),
        }
        for name in META.investigation_features
    }
    DISEASES = list(META.diseases)
    TEST_COSTS = {name: int(m["cost"]) for name, m in MEDICAL_TESTS.items()}
    ASK_COST = META.ask_cost
    return META
