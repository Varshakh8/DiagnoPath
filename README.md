# DiagnoPath AI — RL Clinical Decision Support

DiagnoPath models a clinical consultation as a sequence of decisions using a custom Linear Q-Learning agent. The doctor enters patient findings; the agent selects whether to ask about a symptom, recommend an investigation, or produce a final prediction.

## Run the application (Windows / PowerShell)

From the extracted `DiagnoPath` project folder:

```powershell
python -m pip install -r requirements.txt
python run_app.py
```

Open `http://127.0.0.1:8000`.

Demo login:
- Username: `doctor`
- Password: `clinic123`

A second demo account is also available:
- Username: `dr.mehta`
- Password: `clinic123`

## Single consultation workflow

1. Sign in and choose **Start RL consultation** on the dashboard.
2. Enter an optional patient reference.
3. The RL agent chooses the next symptom question or investigation recommendation.
4. Record the actual symptom answer or investigation result. A test may be explicitly skipped; it is never treated as negative when skipped.
5. The agent updates its observation and chooses the next action or final prediction.
6. SHAP and LIME explanations are generated for the final diagnosis where the packages are available. Feature contribution charts are shown in the consultation.
7. After the final prediction, use **What-if Analysis** to compare the original patient state with a modified hypothetical state. It uses the same RL model and does not alter the saved consultation.

The dataset is retained for model training and evaluation. The old dataset-random consultation and separate live consultation are not exposed as separate user workflows.

## RL model and evaluation

The application loads:
`models/diagnostic_linear_q_warmstart_experiment.json`

The included evaluation script is:
```powershell
python -m rl.evaluate_experiment
```

On the repository's fixed held-out split, the experimental model achieved 34 correct diagnoses out of 42 test cases (81.0%). This is a dataset evaluation result, not clinical validation or a guarantee of accuracy on new patients.

## Clinical feature representation

The existing trained RL model accepts categorical findings:
- `Unknown` = -1
- `Absent / Negative` = 0
- `Present / Positive` = 1

The current model vocabulary does not define numeric laboratory thresholds. Do not enter or interpret raw numeric laboratory values as if this model already understands them. Numeric lab integration requires documented thresholds and compatible preprocessing/model training.

## Explainability and limitations

SHAP and LIME explain feature contributions to the selected RL action's Q-score. Contributions are not causal medical claims, and the normalized relative disease scores displayed by the interface are not calibrated probabilities. The application is an academic decision-support prototype and must not be used as a substitute for professional clinical judgement.

See `IMPLEMENTATION_NOTES.md` for integration details and validation notes.
