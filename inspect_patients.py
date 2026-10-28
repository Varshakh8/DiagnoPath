
from env.diagnostic_env import DiagnosticEnvironment

env = DiagnosticEnvironment()

for i, patient in enumerate(env.dataset.patients):
    if patient.split == "test" and patient.true_label() in (
        "Hepatitis D",
        "Hepatitis E",
    ):
        print("\n" + "=" * 60)
        print("Index:", i)
        print("Disease:", patient.true_label())
        print("Patient object:", patient)

        # Inspect available attributes on the dataset record
        print("Attributes:", vars(patient) if hasattr(patient, "__dict__") else dir(patient))
