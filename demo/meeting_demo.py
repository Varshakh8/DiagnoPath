"""
DiagnoPath AI - Member 1 Meeting Demo
Multi-disease sequential diagnosis using the Kaggle clinical dataset.
No built-in ML algorithms.

Run:
  python demo/meeting_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.data_model import ClinicalDataset, MEDICAL_TESTS, SYMPTOMS, default_data_path
from env.diagnostic_env import Action, ActionType, DiagnosticEnvironment


DEMO_DISEASES = [
    "Malaria",
    "Diabetes",
    "Pneumonia",
    "Fungal infection",
    "Migraine",
]


def banner() -> None:
    print("=" * 68)
    print("  DiagnoPath AI - Member 1 Demo")
    print("  Multi-Disease Clinical Environment (Kaggle dataset)")
    print("  (No built-in ML algorithms)")
    print("=" * 68)


def show_dataset_overview(data: ClinicalDataset) -> None:
    summary = data.summary()
    print("\n[1] REALISTIC MULTI-DISEASE DATASET")
    print(f"    Source : {data.meta.source_name}")
    print(f"    URL    : {data.meta.source_url}")
    print(f"    Patients : {len(data)}")
    print(f"    Diseases : {data.meta.n_diseases}")
    print(f"    Inquiry symptoms : {len(SYMPTOMS)} (cost={data.meta.ask_cost} each)")
    print(
        f"    Investigations    : {len(MEDICAL_TESTS)} "
        f"(cost={data.meta.investigation_cost} each)"
    )
    print("\n    Sample diseases in dataset:")
    for disease in sorted(summary)[:12]:
        print(f"      - {disease}: {summary[disease]} cases")
    print(f"      ... and {max(0, len(summary) - 12)} more diseases")


def scripted_case(env: DiagnosticEnvironment, disease: str) -> None:
    idx = env.dataset.first_index_for(disease)
    patient = env.dataset.get(idx)
    print(f"\n[2] SEQUENTIAL DIAGNOSIS - {disease} (patient {patient.patient_id})")
    obs = env.reset(idx)
    print(f"    {obs.message}")

    # Ask up to 4 present inquiry symptoms (doctor-like path)
    asks = patient.present_symptoms()[:4]
    if not asks:
        asks = list(patient.symptoms.keys())[:3]

    for symptom in asks:
        obs, reward, done, info = env.step(Action(ActionType.ASK_SYMPTOM, symptom))
        print(f"    Step {obs.step}: {obs.message}")
        if done:
            return

    # Order up to 3 present investigations, else first available
    tests = patient.present_investigations()[:3]
    if not tests:
        tests = list(patient.test_values.keys())[:2]

    for test_name in tests:
        obs, reward, done, info = env.step(Action(ActionType.ORDER_TEST, test_name))
        print(f"    Step {obs.step}: {obs.message}")
        if done:
            return

    obs, reward, done, info = env.step(Action(ActionType.DIAGNOSE, disease))
    print(f"    Step {obs.step}: {obs.message}")

    stats = env.episode_stats()
    print("    Episode summary:")
    print(f"      Steps           : {stats['steps']}")
    print(f"      Symptoms asked  : {stats['symptoms_asked']}")
    print(f"      Tests ordered   : {stats['tests_ordered']}")
    print(f"      Total info cost : {stats['total_test_cost']}")
    print(f"      Total reward    : {stats['total_reward']:+.2f}")


def compare_cost_aware(env: DiagnosticEnvironment) -> None:
    print("\n[3] COST-AWARE DECISION POINT (for mentors)")
    print("    Same Malaria patient, two strategies:")

    idx = env.dataset.first_index_for("Malaria")
    patient = env.dataset.get(idx)

    # Strategy A: order many investigations first
    env.reset(idx)
    for t in list(MEDICAL_TESTS.keys())[:12]:
        env.step(Action(ActionType.ORDER_TEST, t))
    env.step(Action(ActionType.DIAGNOSE, "Malaria"))
    a = env.episode_stats()

    # Strategy B: ask present symptoms then few investigations
    env.reset(idx)
    for s in patient.present_symptoms()[:4]:
        env.step(Action(ActionType.ASK_SYMPTOM, s))
    for t in patient.present_investigations()[:2]:
        env.step(Action(ActionType.ORDER_TEST, t))
    env.step(Action(ActionType.DIAGNOSE, "Malaria"))
    b = env.episode_stats()

    print(
        f"    A) Many investigations first -> cost={a['total_test_cost']}, "
        f"reward={a['total_reward']:+.2f}"
    )
    print(
        f"    B) Targeted symptoms + few tests -> cost={b['total_test_cost']}, "
        f"reward={b['total_reward']:+.2f}"
    )
    print("    -> RL (Member 2) will learn efficient multi-disease policies.")


def interactive_mode(env: DiagnosticEnvironment) -> None:
    import random

    print("\n[4] INTERACTIVE MODE (you are the doctor)")
    print("    Open a patient case, ask symptoms / order tests, THEN diagnose.")
    print(f"    Patients available: 0 to {len(env.dataset) - 1}")
    raw_idx = input("    Enter patient index (or press Enter for random): ").strip()
    if raw_idx == "":
        idx = random.randint(0, len(env.dataset) - 1)
    else:
        try:
            idx = int(raw_idx)
        except ValueError:
            print("    Please enter a number, e.g. 150")
            return
        if idx < 0 or idx >= len(env.dataset):
            print("    Patient index out of range.")
            return

    patient = env.dataset.get(idx)
    obs = env.reset(idx)
    print(f"    Opened case {patient.patient_id}. Disease is UNKNOWN - you must diagnose it.")
    print("    Sample symptoms you can ask: chills, vomiting, fatigue, cough, headache, itching")
    print("    Sample tests you can order: high_fever, irregular_sugar_level, rusty_sputum")
    print("    Possible diseases include:", ", ".join(DEMO_DISEASES), ", ...")
    print(
        "    Commands: ask <symptom> | test <investigation> | "
        "diagnose <Disease Name> | quit"
    )

    while not obs.done:
        raw = input("\n    > ").strip()
        if not raw:
            continue
        if raw.lower() in {"quit", "exit", "q"}:
            print("    Exiting interactive mode.")
            return

        parts = raw.split(maxsplit=1)
        if len(parts) != 2:
            print("    Use: ask chills | test high_fever | diagnose Malaria")
            continue

        cmd, target = parts[0].lower(), parts[1].strip()
        try:
            if cmd == "ask":
                action = Action(ActionType.ASK_SYMPTOM, target)
            elif cmd == "test":
                action = Action(ActionType.ORDER_TEST, target)
            elif cmd == "diagnose":
                action = Action(ActionType.DIAGNOSE, target)
            else:
                print("    Unknown command.")
                continue
            obs, reward, done, info = env.step(action)
            print(f"    {obs.message}")
        except Exception as exc:
            print(f"    Error: {exc}")

    print(f"    Final stats: {env.episode_stats()}")


def main() -> None:
    banner()
    data = ClinicalDataset.from_csv(default_data_path())
    show_dataset_overview(data)

    env = DiagnosticEnvironment(data)
    for disease in DEMO_DISEASES[:3]:
        scripted_case(env, disease)

    compare_cost_aware(env)

    print("\n" + "-" * 68)
    choice = input("Run interactive doctor-style mode? [y/N]: ").strip().lower()
    if choice == "y":
        interactive_mode(DiagnosticEnvironment(data))

    print("\nDemo complete. Multi-disease environment is ready for Member 2 (RL).")
    print("=" * 68)


if __name__ == "__main__":
    main()
