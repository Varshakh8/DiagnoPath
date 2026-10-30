# DiagnoPath AI — Consolidated RL Workflow

## Single consultation architecture
The user-facing workflow is one persistent, doctor-entered RL consultation:
1. Doctor starts a consultation and optionally enters a patient reference.
2. The RL agent selects the next symptom inquiry, investigation recommendation, or diagnosis action.
3. Doctor records the actual symptom/test result. A recommended test can be explicitly skipped; skipped tests are not treated as negative.
4. The updated observation is encoded using the existing RL state encoder and passed to the same trained agent.
5. The final prediction and leading disease-score comparison are shown.
6. SHAP and LIME explanations are requested for the final selected diagnosis and rendered as feature-contribution bars.
7. What-if analysis is revealed after the final prediction. It compares original and modified clinical states with the same RL model and shows rankings, next actions, and explanation charts. Simulations do not modify the saved consultation.

## Model used
The application loads `models/diagnostic_linear_q_warmstart_experiment.json`.
The repository's `python -m rl.evaluate_experiment` script was executed against the fixed held-out split: 34 correct of 42 test cases (81.0%). This is a small dataset evaluation result, not clinical validation and not a guarantee for new patients. The model weights were not changed.

## Model input limitation
The current RL encoder accepts categorical feature states: unknown (-1), absent/negative (0), and present/positive (1). The current dataset/model vocabulary does not define numeric laboratory ranges or clinical thresholds. Accordingly, the UI records categorical test results and does not invent numerical cut-offs. Raw numeric lab values require a defined feature preprocessing/threshold specification and model retraining or a compatible encoder update.

## Explainability
The XAI service independently attempts SHAP and LIME. SHAP was executed successfully in the validation environment. LIME is listed in `requirements.txt`, but could not be installed in the offline validation environment; install requirements in the local Python environment to enable it. The interface displays an error for a missing explainer instead of silently claiming it succeeded. Contribution bars explain model Q-scores, not causal medical relationships or calibrated disease probabilities.

## Run
From the extracted project directory, activate the project's Python environment and run:
```powershell
python -m pip install -r requirements.txt
python run_app.py
```
Open `http://127.0.0.1:8000`. Sign in using the existing doctor account, then use **Start RL consultation** on the dashboard.
